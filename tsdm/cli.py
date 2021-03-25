import click
from tsdm.numbers_magic import addition


@click.group()
def cli_group():
    pass


@cli_group.command(help='This command can add numbers')
@click.option(
    '--first-summand',
    help='a first summand',
    required=True,
    type=float
)
@click.option(
    '--second-summand',
    help='the second summand',
    required=True,
    type=float
)
def adding_two_numbers(first_summand, second_summand):
    final_sum = addition(first_summand, second_summand)
    click.echo('The sum is %s' % final_sum)
