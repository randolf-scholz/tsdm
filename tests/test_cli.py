from click.testing import CliRunner
from tsdm.cli import adding_two_numbers


def test_summation_cli():
    # given
    first_summand = 3
    second_summand = 2

    # when
    result = CliRunner().invoke(
        adding_two_numbers,
        [
            '--first-summand', first_summand,
            '--second-summand', second_summand
        ],
    )

    # then
    assert result.exit_code == 0
