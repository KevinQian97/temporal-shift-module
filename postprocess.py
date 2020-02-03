import os
import json
import argparse
import csv

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='prepare results for actev_scorer')
    parser.add_argument('--out_path', dest='out_path',default='/results', type=str,
                      help='output_path')
    parser.add_argument('--prop_path', dest='prop_path',default='/props', type=str,
                      help='prop_path')
    parser.add_argument('--result_path', dest='result_path',default="/results/results.json", type=str,
                      help='prop_path')
    args = parser.parse_args()
    return args




