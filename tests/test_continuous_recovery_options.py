import pytest

from real2sim2real.reconstruction import tracking_recovery


@pytest.mark.parametrize('options', [
    ['--continuous-translation-first'],
    ['--continuous-image-max-size','1536'],
    ['--continuous-cpu-recovery','--continuous-image-max-size','0'],
    ['--continuous-subpixel-radius','1'],
    ['--continuous-cpu-recovery','--continuous-subpixel-radius','nan'],
    ['--continuous-cpu-recovery','--continuous-subpixel-radius','3'],
    ['--continuous-depth-weight', '.5'],
    ['--continuous-cpu-recovery', '--continuous-depth-weight', 'nan'],
    ['--continuous-cpu-recovery', '--continuous-depth-weight', '-1'],
    ['--continuous-cpu-recovery', '--continuous-depth-weight', '6'],
])
def test_invalid_objective_options_fail_before_model_loading(monkeypatch, options):
    def forbidden():
        raise AssertionError('Invalid options must not load GPU models')
    monkeypatch.setattr(tracking_recovery, 'load_exporter', forbidden)
    monkeypatch.setattr('sys.argv', ['tracking_recovery', '--export', 'export',
        '--depth', 'depth', '--seed', 'seed', '--support', 'support',
        '--output', 'output', *options])
    with pytest.raises(SystemExit) as error:
        tracking_recovery.main()
    assert error.value.code == 2
