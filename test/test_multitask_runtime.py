"""マルチタスクモデルおよびビームサーチの最小テスト。"""

from __future__ import annotations

import unittest
from typing import Dict, List

import numpy as np

from beam_search_runtime import select_block_config
from multitask_model import HEAD_ORDER, MultiTaskModel, build_soft_targets


class MultiTaskModelTest(unittest.TestCase):
    """マルチタスクモデル関連のテスト。"""

    def setUp(self) -> None:
        """単純な重み設定のモデルを構築する。"""

        mean = np.zeros(4, dtype=np.float32)
        std = np.ones(4, dtype=np.float32)
        self.model = MultiTaskModel(4, [2], dropout=0.0, feature_mean=mean, feature_std=std)
        self.model.trunk_weights = [
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            )
        ]
        self.model.trunk_biases = [np.zeros(2, dtype=np.float64)]
        for name in HEAD_ORDER:
            classes = self.model.head_weights.get(name)
            if classes is None:
                # 未初期化なので標準的な形状で設定する。
                if name == "predictor":
                    cls = 8
                elif name == "filter_perm":
                    cls = 6
                elif name in ("filter_primary", "filter_secondary"):
                    cls = 4
                elif name == "reorder":
                    cls = 8
                else:
                    cls = 2
                weight = np.zeros((2, cls), dtype=np.float64)
                for i in range(min(2, cls)):
                    weight[i, i] = 1.0
                self.model.head_weights[name] = weight
                self.model.head_biases[name] = np.zeros(cls, dtype=np.float64)
        return super().setUp()

    def test_predict_logits_shapes(self) -> None:
        """predict_logits が各ヘッドの形状を返すか確認する。"""

        features = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        logits = self.model.predict_logits(features)
        self.assertEqual(set(logits.keys()), set(HEAD_ORDER))
        self.assertEqual(logits["predictor"].shape, (8,))
        self.assertEqual(logits["filter_perm"].shape, (6,))
        self.assertEqual(logits["filter_primary"].shape, (4,))
        self.assertEqual(logits["filter_secondary"].shape, (4,))
        self.assertEqual(logits["reorder"].shape, (8,))
        self.assertEqual(logits["interleave"].shape, (2,))

    def test_predict_topk(self) -> None:
        """predict_topk が対数確率順にソートされるか検証する。"""

        features = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        logits = self.model.predict_logits(features)
        topk = self.model.predict_topk(features, k=2)
        predictor_ids = [item[0] for item in topk["predictor"]]
        self.assertEqual(len(predictor_ids), 2)
        self.assertGreaterEqual(logits["predictor"][predictor_ids[0]], logits["predictor"][predictor_ids[1]])

    def test_build_soft_targets(self) -> None:
        """ソフトターゲットの重み付けが正しく行われるかテストする。"""

        best = np.array([1, 2], dtype=np.int32)
        second = np.array([1, 0], dtype=np.int32)
        targets = build_soft_targets(best, second, 4, epsilon=0.2)
        self.assertTrue(np.allclose(targets[0], [0.0, 1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(targets[1], [0.2, 0.0, 0.8, 0.0]))


class BeamSearchTest(unittest.TestCase):
    """ビームサーチ挙動の検証。"""

    def setUp(self) -> None:
        """等確率ロジットを返すモデルを準備する。"""

        mean = np.zeros(4, dtype=np.float32)
        std = np.ones(4, dtype=np.float32)
        self.model = MultiTaskModel(4, [2], dropout=0.0, feature_mean=mean, feature_std=std)
        weight = np.zeros((4, 2), dtype=np.float64)
        self.model.trunk_weights = [weight]
        self.model.trunk_biases = [np.zeros(2, dtype=np.float64)]
        for name in HEAD_ORDER:
            cls = 8 if name in ("predictor", "reorder") else 6 if name == "filter_perm" else 4 if name in ("filter_primary", "filter_secondary") else 2
            self.model.head_weights[name] = np.zeros((2, cls), dtype=np.float64)
            self.model.head_biases[name] = np.zeros(cls, dtype=np.float64)

    def test_beam_cap(self) -> None:
        """候補評価が 16 件以内に制限されることを確認する。"""

        features = np.zeros(4, dtype=np.float32)
        calls: List[Dict[str, int]] = []

        def encoder(config: Dict[str, int]) -> int:
            calls.append(config)
            return config["predictor"] + config["filter"] + config["reorder"] + config["interleave"]

        config, bits = select_block_config(features, self.model, encoder)
        self.assertLessEqual(len(calls), 16)
        self.assertIsInstance(config, dict)
        self.assertIsInstance(bits, int)

    def test_deterministic_selection(self) -> None:
        """同一入力で結果が安定することを確認する。"""

        features = np.zeros(4, dtype=np.float32)

        def encoder(config: Dict[str, int]) -> int:
            return (
                config["predictor"] * 100
                + config["filter"] * 10
                + config["reorder"] * 2
                + config["interleave"]
            )

        result1 = select_block_config(features, self.model, encoder)
        result2 = select_block_config(features, self.model, encoder)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
