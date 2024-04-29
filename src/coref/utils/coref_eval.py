import subprocess
import re
from argparse import ArgumentParser
import logging

logger = logging.getLogger(__name__)

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["../conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-g", "--gold_file")
    parser.add_argument("-p", "--pred_file")
    args = parser.parse_args()

    gold_file = args.gold_file
    pred_file = args.pred_file
    results = {m: official_conll_eval(gold_file, pred_file, m) for m in ("muc", "bcub", "ceafe")}
    print(results)
