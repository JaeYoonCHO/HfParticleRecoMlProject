"""Run with: python -m unittest discover -s model_codes/tests -v"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import uproot
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Inference.Inference import build_particle_dict_from_root, InferenceDataset, predict_prob1
from Inference.batch_inference import build_bins
from data_preprocessing import get_particle_feature_map
from data_loader import ParticleDataset
from model import ParticleTransformerClassifier
from particle_embedding import ParticleEmbedding
from train import check_finite, evaluate_one_epoch, main


class ConfigurationTests(unittest.TestCase):
    def test_nonfinite_values_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaisesRegex(ValueError, 'NaN or Inf'):
                check_finite(torch.tensor([value]), 'logits', 'train, epoch 1')
        check_finite(torch.tensor([0, 1]), 'labels', 'train')

    def test_csv_particle_selection_without_description(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'particles.csv'
            path.write_text('Branch,Particle\nb,parent\na,daughter\nc,parent\n')
            self.assertEqual(get_particle_feature_map(path), {'parent': ['b', 'c'], 'daughter': ['a']})
            self.assertEqual(get_particle_feature_map(path, {
                'child_token': {'particle': 'daughter'}, 'parent_token': ['c', 'b'],
            }), {'child_token': ['a'], 'parent_token': ['c', 'b']})
            with self.assertRaises(ValueError):
                get_particle_feature_map(path, {'unknown': {'particle': 'absent'}})
        self.assertEqual(get_particle_feature_map(None, {'any_chain': ['f1']}), {'any_chain': ['f1']})
        with self.assertRaises(ValueError):
            get_particle_feature_map(None, {'bad': ['f1', 'f1']})

    def test_positional_encoding_and_state_compatibility(self):
        inputs = {'parent': torch.randn(5, 2), 'daughter': torch.randn(5, 3)}
        disabled = ParticleEmbedding(inputs, embed_dim=8, dropout=0).eval()
        enabled = ParticleEmbedding(inputs, embed_dim=8, dropout=0, use_positional_encoding=True).eval()
        enabled.load_state_dict(disabled.state_dict(), strict=True)
        tokens = torch.stack([disabled.embedders[k](v) for k, v in inputs.items()], dim=1)
        tokens = torch.cat([disabled.cls_token.expand(5, -1, -1), tokens], dim=1)
        torch.testing.assert_close(disabled(inputs), disabled.layer_norm(tokens), rtol=0, atol=0)
        torch.testing.assert_close(enabled(inputs), enabled.layer_norm(tokens + enabled.pos_encoding))
        self.assertFalse(torch.equal(enabled(inputs), disabled(inputs)))
        model = ParticleTransformerClassifier(inputs, embed_dim=8, num_heads=2, num_layers=1,
                                              ff_dim=16, use_positional_encoding=True).eval()
        self.assertTrue(model.embedding.use_positional_encoding)
        self.assertEqual(model(inputs).shape, (5, 2))

    def test_train_eval_covers_all_rows_without_changing_weights_or_rng(self):
        inputs = {'generic': torch.randn(7, 2)}
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0])
        dataset = ParticleDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=False,
                            generator=torch.Generator().manual_seed(17))
        model = ParticleTransformerClassifier(inputs, embed_dim=8, num_heads=2, num_layers=1,
                                              ff_dim=16, dropout=0.5)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        summary, predictions, _ = evaluate_one_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                                                      torch.device('cpu'), 1, 1, 'train_eval')
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(summary['num_samples'], 7)
        self.assertEqual(predictions['index'].tolist(), list(range(7)))
        self.assertAlmostEqual(summary['auc'], roc_auc_score(labels, predictions['prob_1']))
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)

    def test_training_saves_train_eval_history_with_explicit_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'f1': np.arange(9, dtype=np.float32),
                    'f2': np.arange(9, dtype=np.float32) ** 2,
                    'isSignal': np.arange(9, dtype=np.int64) % 2}
            data_path = root / 'input.root'
            with uproot.recreate(data_path) as f:
                f.mktree('OtherDecayTree', data)
            cfg = {
                'paths': {**{k: str(data_path) for k in ('train_csv', 'val_csv', 'test_csv')},
                          'stats_save_path': str(root / 'stats.pt')},
                'particles': {'mother': ['f2'], 'child': ['f1']},
                'data': {'tree_name': 'OtherDecayTree', 'batch_size': 4, 'num_workers': 0, 'pin_memory': False,
                         'label_column': 'isSignal', 'drop_last': True},
                'model': {'embed_dim': 8, 'num_heads': 2, 'num_layers': 1, 'ff_dim': 16,
                          'dropout': 0.1, 'num_classes': 2, 'head_hidden_dim': 8,
                          'use_final_norm': True, 'use_positional_encoding': True},
                'train': {'seed': 42, 'device': 'cpu', 'epochs': 2,
                          'learning_rate': 0.001, 'weight_decay': 0.0001},
                'scheduler': {'use_scheduler': False},
                'metrics': {'score_threshold': 0.6},
                'save': {'save_dir': str(root / 'out'), 'best_model_name': 'best.pt',
                         'final_model_name': 'final.pt'},
                'log': {'print_every': 1},
            }
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            with contextlib.redirect_stdout(io.StringIO()):
                main(str(config_path))
            history = pd.read_csv(root / 'out/history/training_history.csv')
            self.assertEqual(history['split'].tolist(), ['train', 'val', 'train_eval'] * 2)
            self.assertEqual(history.loc[history.split == 'train', 'num_samples'].tolist(), [8, 8])
            self.assertEqual(history.loc[history.split == 'train_eval', 'num_samples'].tolist(), [9, 9])
            self.assertTrue(history['auc'].between(0, 1).all())
            self.assertNotIn('ratio', history.columns)
            self.assertTrue((history['score_threshold'] == 0.6).all())
            for metric in ('purity', 'signal_efficiency', 'background_efficiency', 'background_rejection'):
                self.assertTrue(history[metric].dropna().between(0, 1).all())
            test_metrics = pd.read_csv(root / 'out/history/test_summary.csv')
            self.assertEqual(test_metrics['score_threshold'].iloc[0], 0.6)
            for frame in (history, test_metrics):
                self.assertTrue((frame.num_signal + frame.num_background == frame.num_samples).all())
                defined = frame.selected_signal + frame.selected_background > 0
                np.testing.assert_allclose(
                    frame.loc[defined, 'purity'],
                    frame.loc[defined, 'selected_signal'] /
                    (frame.loc[defined, 'selected_signal'] + frame.loc[defined, 'selected_background']))
            checkpoint = torch.load(root / 'out/final_checkpoint.pt', weights_only=False)
            self.assertEqual(checkpoint['config']['particles'], cfg['particles'])
            self.assertTrue(checkpoint['config']['model']['use_positional_encoding'])
            stats = torch.load(root / 'stats.pt', weights_only=False)
            _, inputs, mask = build_particle_dict_from_root(
                str(data_path), 'OtherDecayTree', None, stats, cfg['particles'])
            self.assertTrue(mask.all())
            model = ParticleTransformerClassifier(inputs, **cfg['model']).eval()
            model.load_state_dict(checkpoint['model_state_dict'])
            loader = DataLoader(InferenceDataset(inputs), batch_size=4)
            single_scores = predict_prob1(model, loader, torch.device('cpu'))
            batch_cfg = {'model': cfg['model'], 'pt_bins': [{
                'name': 'all', 'pt_min': 0, 'pt_max': 100,
                'stats_path': str(root / 'stats.pt'),
                'model_path': str(root / 'out/final.pt'),
            }]}
            bins = build_bins(batch_cfg, cfg['particles'], torch.device('cpu'), torch.float32)
            self.assertTrue(bins[0]['model'].embedding.use_positional_encoding)
            with torch.no_grad():
                batch_scores = bins[0]['model'](inputs).softmax(-1)[:, 1].numpy()
            np.testing.assert_allclose(single_scores, batch_scores, rtol=1e-5, atol=1e-6)



if __name__ == '__main__':
    unittest.main()
