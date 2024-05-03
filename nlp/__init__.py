# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """Zhige Wang"""
__email__ = 'zwang64@tulane.edu'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
    return
    # w = open(path, 'wt')
    # w.write('[data]\n')
    # w.write('urlfake = https://tulane.app.box.com/s/yqd2meu8exi6b66zp668nyahd1i2msvs?dl=1\n')
    # w.write('urltrue = https://tulane.box.com/s/sqek0ugmvst7bysz7bfpm5s0pm6xs655?dl=1\n')
    # w.write('filefake = %s%s%s\n' % (nlp_path, os.path.sep, 'fake.csv'))
    # w.write('filetrue = %s%s%s\n' % (nlp_path, os.path.sep, 'true.csv'))
    # w.close()

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['HOME'] + os.path.sep + '.nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
model_path = nlp_path + 'model.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)