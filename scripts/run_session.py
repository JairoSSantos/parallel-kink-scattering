import sys
sys.path.insert(1, '../')

from dataclasses import fields, MISSING
from modules.parallel import CollisionSession
import argparse

parser = argparse.ArgumentParser(
    prog='RunSession',
    description='Used to init the collision sessions.',
)
for field in fields(CollisionSession):
    parser.add_argument(('' if field.default == MISSING else '--') + field.name, default=field.default, type=field.type)

if __name__ == '__main__':
    session = CollisionSession(**vars(parser.parse_args()))
    session.run()