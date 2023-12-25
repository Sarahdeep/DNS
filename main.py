import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shlex
import glob


def get_argument(cmd, s, default):
    try:
        val = cmd[cmd.index(s) + 1]
    except ValueError:
        val = default
    return val


def prepare_df(filepath):
    df = pd.read_json(filepath, lines=True)
    cmd = shlex.split(df.iloc[0, :]["cmdline"])
    df = df[pd.isna(df['total_response_avg_ms'])][1:]
    return df, cmd


def get_test(filepath):
    tests = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8']
    return [x for x in tests if x in filepath][0]


def get_resolver(cmd, resolvers):
    return [x for x in resolvers if x in cmd][0]


def get_protocol(cmd):
    protocol = get_argument(cmd, '-P', 'udp')
    if protocol == 'doh':
        method = get_argument(cmd, '-M', 'GET')
        protocol += '-' + method
    return protocol


def parse_files(directories, subdir):
    files = {}
    for directory in directories:
        for filepath in sorted(glob.glob(directory + '/**/*.json', recursive=True)):
            if '\\' + subdir + '\\' in filepath:
                df, cmd = prepare_df(filepath)
                protocol = get_protocol(cmd)
                record = df.to_records().reshape(-1, 1)
                test = get_test(filepath)
                if (protocol, test) not in files:
                    files[(protocol, test)] = [record]
                else:
                    files[(protocol, test)].append(record)
    return files


def get_data_and_merge(files, data_extractor, post_extractor=lambda x: x):
    extracted_values = {}
    for key, values in files.items():
        val = data_extractor(values)
        if isinstance(val, np.ndarray) and val.size == 0:
            continue
        val = post_extractor(val)
        if len(key) > 2:
            current_key = (key[0], key[2])
        else:
            current_key = key[0]
        if current_key not in extracted_values:
            extracted_values[current_key] = [val]
        else:
            extracted_values[current_key].append(val)
    return extracted_values


def filt(val):
    return bool(val) and not np.isnan(val)


def get_rtt(values):
    return np.concatenate(
        [np.concatenate(list(filter(filt, x.period_response_avg_ms))) for x in values]).flatten()


def rtt_mean_plot(files, subdir):
    rtt_mean = get_data_and_merge(files, get_rtt, np.mean)
    three_bar_plot(rtt_mean['udp'], rtt_mean['doh-GET'], rtt_mean['doh-POST'],
                   subdir + '-' + 'combined-rtt-mean.png', "Тесты определены в таблице",
                   "Round-trip time in milliseconds")
    difference_bar_plot(rtt_mean['udp'], rtt_mean['doh-GET'], rtt_mean['doh-POST'],
                        subdir + '-' + 'combined-rtt-difference.png', "Тесты определены в таблице",
                        "Round-trip time difference (UDP) in milliseconds")


def three_bar_plot(array_udp, array_doh_get, array_doh_post, filename, xlabel, ylabel, width=5):
    x_pos = np.arange(1, len(array_udp) + 1)
    plt.figure()
    plt.rc('font', size=11)
    plt.rc('legend', fontsize=12)
    plt.bar((x_pos * width * 4) - width, array_udp, width=width, color='blue', label="UDP", align='center')
    plt.bar((x_pos * width * 4), array_doh_get, width=width, color='orange', label="HTTPS GET",
            align='center')
    plt.bar((x_pos * width * 4) + width, array_doh_post, width=width, color='green',
            label="HTTPS POST", align='center')
    plt.xticks(x_pos * width * 4, ["Test " + str(x) for x in x_pos])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    print("Saved {}".format(filename))


def bar_plot(handshakes, filename, xlabel, ylabel, stepsize=10, width=5):
    x_pos = np.arange(1, len(handshakes) + 1)
    plt.figure()
    plt.rc('font', size=11)
    plt.rc('legend', fontsize=12)
    bars = plt.bar((x_pos * width * 2), handshakes, width=width, color='blue', align='center')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    for bar in bars:
        bar_height = bar.get_height()
        plt.text(bar.get_x() - width / 4, bar_height + 2, format(bar_height, '.2f'))

    plt.xticks(x_pos * width * 2, ["Test " + str(x) for x in x_pos])
    y_ticks = np.arange(0, round(max(handshakes) + stepsize * 2, -1), stepsize)
    plt.yticks(y_ticks)
    plt.savefig(filename)
    print("Saved {}".format(filename))


def difference_bar_plot(array_udp, array_doh_get, array_doh_post, filename, xlabel, ylabel, width=5):
    x_pos = np.arange(1, len(array_udp) + 1)
    plt.figure()
    plt.rc('font', size=11)
    plt.rc('legend', fontsize=12)
    array_doh_post_difference = np.subtract(np.asarray(array_doh_post), np.asarray(array_udp))
    array_doh_get_difference = np.subtract(np.asarray(array_doh_get), np.asarray(array_udp))
    plt.axhline(y=0, color='black', linestyle='--')
    plt.bar((x_pos * width * 4), array_doh_get_difference, width=width, color='orange',
            label="HTTPS GET", align='center')
    plt.bar((x_pos * width * 4) + width, array_doh_post_difference, width=width, color='green',
            label="HTTPS POST", align='center')
    plt.xticks(x_pos * width * 4, ["Test " + str(x) for x in x_pos])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    print("Saved {}".format(filename))


if __name__ == "__main__":
    subdir = 'tests'
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', nargs='+', metavar='DIRS', help="A list of directories to use as input",
                        required=True)
    args = parser.parse_args()
    files = parse_files(args.dirs, subdir)
    rtt_mean_plot(files, subdir)
