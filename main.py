import click

from cli.manage import cli_manage
from cli.measure import cli_measure
from cli.singlepredict import cli_predict

# Collect command groups
cli = click.CommandCollection(sources=[cli_predict,
        cli_manage, cli_measure],
        help="""Commands which perform skin detection (batch, bench, image, single)
        need to be executed with the appended string\n
        ' | sudo docker-compose exec -T opencv bash'\n
        eg.\n
        python main.py single -t ECU | sudo docker-compose exec -T opencv bash""")

if __name__ == "__main__":
    # Setup Command Line Interface
    # Register commands
    cli()
