import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.append(Path(__file__).parents[1].as_posix())

from los_estimator.config import default_config_path
from los_estimator.estimation_run import LosEstimationRun, load_configurations
from util.comparison_data_loader import load_comparison_data


class TestLosEstimatorIntegration:
    """Integration tests for LOS Estimator."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Set up test data and configurations."""
        self.original_result = load_comparison_data(less_windows=True)
        self.cfg = load_configurations(default_config_path)

        # Configure for testing
        self.cfg["visualization_config"].show_figures = False
        self.cfg["animation_config"].show_figures = False
        self.cfg["animation_config"].debug_animation = True
        self.cfg["debug_config"].one_window = False
        self.cfg["debug_config"].less_windows = True
        self.cfg["debug_config"].less_distros = False
        self.cfg["debug_config"].only_linear = False

    def test_estimation_run_completes_successfully(self):
        """Test that the estimation run completes without errors."""
        estimator = self._run_estimator()

        # Basic assertions
        assert hasattr(estimator, "all_fit_results")
        assert estimator.all_fit_results is not None
        assert len(estimator.all_fit_results) > 0

    def test_all_expected_distributions_present(self):
        """Test that all expected distributions are present in results."""
        estimator = self._run_estimator()
        new_result = estimator.all_fit_results

        for distro in self.original_result.keys():
            assert (
                distro in new_result
            ), f"Distribution {distro} not found in new results"

    def test_kernel_comparison_accuracy(self):
        """Test that kernel values match expected results within tolerance."""
        estimator = self._run_estimator()
        new_result = estimator.all_fit_results

        for distro, fit_result in new_result.items():
            if distro == "compartmental" or distro not in self.original_result:
                continue

            comp_fit_result = self.original_result[distro]
            kernel_diff = np.abs(
                fit_result.all_kernels - comp_fit_result.all_kernels
            ).max()

            assert np.allclose(
                fit_result.all_kernels, comp_fit_result.all_kernels, atol=1e-4
            ), f"Kernel comparison failed for {distro}. Max difference: {kernel_diff:.4f}"

    def test_error_metrics_accuracy(self):
        """Test that train and test error metrics match expected values."""
        estimator = self._run_estimator()
        new_result = estimator.all_fit_results

        for distro, fit_result in new_result.items():
            if distro == "compartmental" or distro not in self.original_result:
                continue

            comp_fit_result = self.original_result[distro]

            train_error_diff = np.abs(
                fit_result.train_relative_errors.mean()
                - comp_fit_result.train_relative_errors.mean()
            )
            test_error_diff = np.abs(
                fit_result.test_relative_errors.mean()
                - comp_fit_result.test_relative_errors.mean()
            )

            assert (
                train_error_diff <= 1e-4
            ), f"Train error difference too large for {distro}: {train_error_diff:.4f}"
            assert (
                test_error_diff <= 1e-4
            ), f"Test error difference too large for {distro}: {test_error_diff:.4f}"

    def test_cli_execution_completes_successfully(self):
        """Test that the CLI execution completes without errors."""
        # Assuming there's a main CLI script, adjust path as needed
        cli_script = Path(__file__).parents[1] / "los_estimator/__main__.py"

        if not cli_script.exists():
            pytest.skip("CLI script not found, skipping CLI test")

        # Run the CLI with test parameters
        cmd = [
            sys.executable,
            str(cli_script),
            "--config",
            str(default_config_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )

            # Print the output for debugging purposes
            print("CLI stdout:", result.stdout)
            print("CLI stderr:", result.stderr)

            # Assertions
            assert (
                result.returncode == 0
            ), f"CLI execution failed with return code {result.returncode}"
            # if result.stderr is not None:
            #     assert "error" not in result.stderr.lower(), f"CLI stderr contains errors: {result.stderr}"

            # # Check for expected output patterns
            # assert any(
            #     keyword in result.stdout.lower() for keyword in ["completed", "success", "finished"]
            # ), "CLI output doesn't indicate successful completion"

        except subprocess.TimeoutExpired:
            pytest.fail("CLI execution timed out after 5 minutes")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"CLI execution failed: {e.stderr}")

    def _run_estimator(self):
        """Helper method to run the estimator with test configuration."""
        unique_id = str(uuid.uuid4())
        estimator = LosEstimationRun(
            self.cfg["data_config"],
            self.cfg["output_config"],
            self.cfg["model_config"],
            self.cfg["debug_config"],
            self.cfg["visualization_config"],
            self.cfg["animation_config"],
            run_nickname=f"test_run_{unique_id}",
        )
        estimator.run_analysis(vis=False)
        return estimator


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    # TestLosEstimatorIntegration().test_cli_execution_completes_successfully()
