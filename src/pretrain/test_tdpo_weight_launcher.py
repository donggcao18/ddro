"""Check the four-V100 launcher without loading data or starting GPU processes."""

import os
from pathlib import Path
import shutil
import subprocess
import unittest


class WeightLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [Path("C:/Program Files/Git/bin/bash.exe")] if os.name == "nt" else []
        cls.bash = next((str(path) for path in candidates if path.is_file()), shutil.which("bash"))
        if not cls.bash:
            raise unittest.SkipTest("Bash is required for launcher checks")
        cls.root = Path(__file__).resolve().parents[2]
        cls.script = "src/scripts/ddro/launch_tdpo2_weight_vault_url_4v100.sh"

    def run_launcher(self, **overrides):
        env = os.environ.copy()
        for key in [
            "CUDA_VISIBLE_DEVICES", "NUM_GPUS", "PREFERENCE_OBJECTIVE", "TDPO_PREFIX_TOKENS",
            "TDPO_PREFIX_WEIGHT", "TRAIN_BATCH_SIZE", "EVAL_BATCH_SIZE",
            "GRADIENT_ACCUMULATION_STEPS", "GRADIENT_CHECKPOINTING", "PRECISION", "OUTPUT_DIR",
            "LOG_DIR", "RESUME_FROM_CHECKPOINT", "PRINT_ONLY",
        ]:
            env.pop(key, None)
        # Select the same Bash for the wrapper's exec, including on Windows.
        env["PATH"] = str(Path(self.bash).parent) + os.pathsep + env.get("PATH", "")
        env.update(PRINT_ONLY="1", **overrides)
        return subprocess.run([self.bash, self.script], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=20)

    def test_default_command(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--nproc_per_node=4", result.stdout)
        self.assertIn("--preference_objective tdpo2-weight", result.stdout)
        self.assertIn("--tdpo_prefix_tokens 3", result.stdout)
        self.assertIn("--tdpo_prefix_weight 3.0", result.stdout)
        self.assertIn("effective batch size: 32", result.stdout)
        self.assertIn("Resume: none", result.stdout)
        self.assertNotIn("--resume_from_checkpoint", result.stdout)
        self.assertNotIn("--bf16", result.stdout)
        self.assertNotIn("--fp16", result.stdout)

    def test_scheduler_visibility_and_experiment_overrides(self):
        result = self.run_launcher(CUDA_VISIBLE_DEVICES="4,5,6,7", TDPO_PREFIX_TOKENS="2",
                                   TDPO_PREFIX_WEIGHT="5", PRECISION="fp16",
                                   RESUME_FROM_CHECKPOINT="latest")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPUs: 4,5,6,7", result.stdout)
        self.assertIn("--tdpo_prefix_tokens 2", result.stdout)
        self.assertIn("--tdpo_prefix_weight 5", result.stdout)
        self.assertIn("tdpo2-weight-k2-w5-4v100", result.stdout)
        self.assertIn("--fp16", result.stdout)
        self.assertIn("--resume_from_checkpoint", result.stdout)

    def test_invalid_precision_or_gpu_assignment_fails(self):
        for overrides in [{"PRECISION": "bf16"}, {"NUM_GPUS": "2"},
                          {"CUDA_VISIBLE_DEVICES": "0,1"}]:
            with self.subTest(overrides=overrides):
                result = self.run_launcher(**overrides)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
