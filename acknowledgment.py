import numpy as np
import matplotlib.pyplot as plt
import os
import dpkt
from concurrent.futures import ProcessPoolExecutor
import json
import csv

plt.rcParams.update({
    'font.size': 18, 
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'lines.linewidth': 2,
}) 
plt.rcParams['font.family'] = 'STIXGeneral'

SEQUENCES_DIR = "sequences"
RESULTS_DIR = "results"
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def format_value_for_filename(value):
    return str(value).replace('.', '_')


def get_sequence_filename(distribution, n, trials, seed, **kwargs):
    if distribution == 'MAWI':
        pcap_file = kwargs.get('pcap_file', 'data')
        pcap_base = os.path.splitext(os.path.basename(pcap_file))[0]
        target_mean = kwargs.get('target_mean', 0.2)
        target_mean_str = format_value_for_filename(target_mean)
        filename = f"seq_MAWI_{pcap_base}_n{n}_trials{trials}_seed{seed}_mean{target_mean_str}.npz"
    elif distribution == 'Pareto':
        alpha = kwargs.get('alpha', 3)
        scale = kwargs.get('scale', 0.5)
        alpha_str = format_value_for_filename(alpha)
        scale_str = format_value_for_filename(scale)
        filename = f"seq_Pareto_n{n}_trials{trials}_seed{seed}_alpha{alpha_str}_scale{scale_str}.npz"
    elif distribution == 'Poisson':
        scale = kwargs.get('scale', 0.3)
        scale_str = format_value_for_filename(scale)
        filename = f"seq_Poisson_n{n}_trials{trials}_seed{seed}_scale{scale_str}.npz"
    else:
        filename = f"seq_{distribution}_n{n}_trials{trials}_seed{seed}.npz"

    return os.path.join(SEQUENCES_DIR, filename)


def save_sequences(arrivals_all, filename, metadata):
    np.savez_compressed(
        filename,
        arrivals=np.array(arrivals_all, dtype=object),
        metadata=json.dumps(metadata)
    )
    print(f"Sequences saved to {filename}")


def load_sequences(filename):
    if os.path.exists(filename):
        print(f"Loading cached sequences from {filename}")
        data = np.load(filename, allow_pickle=True)
        arrivals_all = data['arrivals'].tolist()
        metadata = json.loads(str(data['metadata']))
        return arrivals_all, metadata
    return None, None


def check_sequence_exists(distribution, n, trials, seed, **kwargs):
    filename = get_sequence_filename(distribution, n, trials, seed, **kwargs)
    return os.path.exists(filename)


def get_or_extract_timestamps(pcap_file, npy_file=None):
    if npy_file is None:
        pcap_base = os.path.splitext(os.path.basename(pcap_file))[0]
        npy_file = os.path.join(SEQUENCES_DIR, f"timestamps_{pcap_base}.npy")

    if os.path.exists(npy_file):
        print(f"Loading cached timestamps from {npy_file}")
        return np.load(npy_file)

    print(f"Extracting timestamps from {pcap_file}...")
    timestamps = []
    with open(pcap_file, 'rb') as f:
        reader = dpkt.pcap.Reader(f)
        for ts, _ in reader:
            timestamps.append(ts)
    timestamps = np.unique(np.array(timestamps))
    np.save(npy_file, timestamps)
    print(f"Timestamps saved to {npy_file}")
    return timestamps


def compute_trial_data(args):
    i, arrivals, predictions, lambda_values, rho = args

    def optimal_offline(arrivals: list[int], rho: float):
        n = len(arrivals)
        arrivals = sorted(arrivals)
        _dp_par = [-1]
        dp = [0]

        for i in range(n):
            dp.append(rho + dp[-1])
            _dp_par.append(i)
            delay_max = 0
            for j in range(i, -1, -1):
                delay_max = max(delay_max, arrivals[i]-arrivals[j])
                tmp = dp[j] + delay_max * (1-rho) + rho
                if dp[-1] > tmp:
                    dp[-1] = tmp
                    _dp_par[-1] = j

        return dp[-1]

    def classic_deterministic(arrivals, rho):
        arrivals = sorted(arrivals)
        threshold = rho / (1 - rho)
        acks = []
        pending = []

        i = 0
        n = len(arrivals)
        cost = 0

        while i < n:
            a = arrivals[i]
            pending.append(a)
            deadline = pending[-1] + threshold

            next_arrival = arrivals[i + 1] if i + 1 < n else float('inf')

            if next_arrival > deadline:
                delay = deadline - pending[0]
                cost += rho + delay * (1 - rho)
                acks.append(deadline)
                pending = []
            i += 1

        if pending:
            last_time = arrivals[-1]
            delay = last_time - pending[0]
            cost += rho + delay * (1 - rho)
            acks.append(last_time)

        return cost

    def ack_after_each(arrivals, rho):
        n = len(arrivals)
        return n * rho

    def fixed_time_ack(arrivals, T, rho):
        arrivals = sorted(arrivals)
        cost = 0.0
        pending = []
        i = 0
        n = len(arrivals)

        current_ack_time = T

        while i < n:
            a = arrivals[i]

            while a > current_ack_time:
                if pending:
                    delay = current_ack_time - pending[0]
                    cost += rho + delay * (1 - rho)
                    pending = []
                current_ack_time += T
            pending.append(a)
            i += 1

        if pending:
            delay = current_ack_time - pending[0]
            cost += rho + delay * (1 - rho)

        return cost

    def trust_with_prediction(arrivals, predictions, rho):
        arrivals = sorted(arrivals)
        threshold = rho / (1 - rho)
        acks = []
        pending = []

        i = 0
        n = len(arrivals)
        cost = 0

        while i < n:
            a = arrivals[i]
            pending.append(a)

            pred_next = predictions[i + 1] if i + 1 < n else float('inf')
            next_arrival = arrivals[i + 1] if i + 1 < n else float('inf')

            if pred_next - arrivals[i] > threshold:
                deadline = arrivals[i]
            else:
                deadline = max(arrivals[i], pred_next + threshold)

            if next_arrival > deadline:
                delay = deadline - pending[0]
                cost += rho + delay * (1 - rho)
                acks.append(deadline)
                pending = []
            i += 1

        if pending:
            last_time = arrivals[-1]
            delay = last_time - pending[0]
            cost += rho + delay * (1 - rho)
            acks.append(last_time)

        return cost

    def augmented_algorithm(arrivals, predictions, rho, lambda_param):
        arrivals = sorted(arrivals)
        threshold = rho / (1 - rho)
        acks = []
        pending = []

        i = 0
        n = len(arrivals)
        cost = 0

        while i < n:
            a = arrivals[i]
            pending.append(a)

            pred_next = predictions[i + 1] if i + 1 < n else float('inf')
            next_arrival = arrivals[i + 1] if i + 1 < n else float('inf')

            if pred_next - arrivals[i] > threshold:
                deadline = arrivals[i] + (lambda_param) * threshold
            else:
                deadline = arrivals[i] + (1/lambda_param) * threshold

            if next_arrival > deadline:
                delay = deadline - pending[0]
                cost += rho + delay * (1 - rho)
                acks.append(deadline)
                pending = []
            i += 1

        if pending:
            last_time = arrivals[-1]
            delay = last_time - pending[0]
            cost += rho + delay * (1 - rho)
            acks.append(last_time)

        return cost

    if len(arrivals) > 1:
        mean_interarrival = float(np.mean(np.diff(sorted(arrivals))))
    else:
        mean_interarrival = 0.0

    opt = optimal_offline(arrivals, rho)
    result = {
        "classic": calculate_competitive_ratio(classic_deterministic(arrivals[:], rho), opt),
        "trust": calculate_competitive_ratio(trust_with_prediction(arrivals[:], predictions[:], rho), opt),
        "ack_each": calculate_competitive_ratio(ack_after_each(arrivals[:], rho), opt),
        "fixed_time": calculate_competitive_ratio(fixed_time_ack(arrivals[:], mean_interarrival, rho), opt),
    }
    for lam in lambda_values:
        result[lam] = calculate_competitive_ratio(augmented_algorithm(arrivals[:], predictions[:], rho, lam), opt)
    return result


def generate_arrivals(n, distribution='Pareto', alpha=3, scale=0.5):
    if distribution == 'Poisson':
        inter_arrivals = np.random.exponential(scale=scale, size=n)
    elif distribution == 'Pareto':
        inter_arrivals = np.random.pareto(alpha, size=n) * scale
    else:
        raise ValueError("Unknown distribution")
    arrivals = np.cumsum(inter_arrivals)
    return arrivals


def get_sample_trials(timestamps, n=50, trials=1000, seed=43, target_mean=0.2):
    np.random.seed(seed)
    timestamps = np.asarray(timestamps)
    max_start = len(timestamps) - n
    starts = np.random.randint(0, max_start, size=trials)

    trials_arrivals = []
    for start in starts:
        segment = timestamps[start:start + n]
        segment -= segment[0]
        inter_arrivals = np.diff(segment, prepend=0)
        scale = inter_arrivals.mean()
        normalized = segment * (target_mean / scale)
        trials_arrivals.append(normalized.tolist())

    return trials_arrivals


def get_or_generate_sequences(distribution, n, trials, seed, **kwargs):
    filename = get_sequence_filename(distribution, n, trials, seed, **kwargs)

    arrivals_all, metadata = load_sequences(filename)
    if arrivals_all is not None:
        return arrivals_all

    print(f"Generating new sequences for {distribution}...")

    if distribution == 'MAWI':
        timestamps = kwargs.get('timestamps')
        target_mean = kwargs.get('target_mean', 0.2)
        if timestamps is None:
            raise ValueError("timestamps required for MAWI distribution")
        arrivals_all = get_sample_trials(timestamps, n, trials, seed, target_mean)
        metadata = {
            'distribution': distribution,
            'n': n,
            'trials': trials,
            'seed': seed,
            'target_mean': target_mean,
            'pcap_file': kwargs.get('pcap_file', '')
        }
    elif distribution == 'Pareto':
        alpha = kwargs.get('alpha', 3)
        scale = kwargs.get('scale', 0.5)
        np.random.seed(seed)
        arrivals_all = [generate_arrivals(n, distribution, alpha, scale).tolist() 
                        for _ in range(trials)]
        metadata = {
            'distribution': distribution,
            'n': n,
            'trials': trials,
            'seed': seed,
            'alpha': alpha,
            'scale': scale
        }
    elif distribution == 'Poisson':
        scale = kwargs.get('scale', 0.3)
        np.random.seed(seed)
        arrivals_all = [generate_arrivals(n, distribution, scale=scale).tolist() 
                        for _ in range(trials)]
        metadata = {
            'distribution': distribution,
            'n': n,
            'trials': trials,
            'seed': seed,
            'scale': scale
        }
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    save_sequences(arrivals_all, filename, metadata)

    return arrivals_all


def add_prediction_error(arrivals, error_level):
    n = len(arrivals)
    total_error = error_level * n 
    errors = np.random.uniform(0, 1, size=n)
    errors = errors / np.sum(errors) * total_error
    signs = np.random.choice([-1, 1], size=n)
    predictions = [max(0, a + e * s) for a, e, s in zip(arrivals, errors, signs)]
    return predictions


def calculate_competitive_ratio(algo_cost, opt_cost):
    if opt_cost == 0:
        return float('inf')
    return algo_cost / opt_cost


def save_results_to_csv(error_levels, results_dict, distribution, lambda_values):
    csv_filename = os.path.join(RESULTS_DIR, f"results_{distribution}.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['error_level', 'classic', 'trust', 'ack_each', 'fixed_time']
        fieldnames.extend([f'lambda_{format_value_for_filename(lam)}' for lam in lambda_values])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, error in enumerate(error_levels):
            row = {
                'error_level': f'{error:.6f}',
                'classic': f'{results_dict["classic"][i]:.6f}',
                'trust': f'{results_dict["trust"][i]:.6f}',
                'ack_each': f'{results_dict["ack_each"][i]:.6f}',
                'fixed_time': f'{results_dict["fixed_time"][i]:.6f}',
            }
            for lam in lambda_values:
                lam_key = format_value_for_filename(lam)
                row[f'lambda_{lam_key}'] = f'{results_dict[lam][i]:.6f}'

            writer.writerow(row)

    print(f"Results saved to {csv_filename}")
    return csv_filename


def plot_competitive_ratio_vs_errors(rho, lambda_values, distribution, alpha=3, scale=0.5, 
                                     timestamps=None, pcap_file=None, n=100, trials=100, seed=43):
    error_levels = np.linspace(0, 1.5, 50)
    classic_stats = []
    trust_stats = []
    ack_each_stats = []
    fixed_time_stats = []
    augmented_stats = {lam: [] for lam in lambda_values}

    kwargs = {'alpha': alpha, 'scale': scale}
    if distribution == 'MAWI':
        kwargs['timestamps'] = timestamps
        kwargs['pcap_file'] = pcap_file
        kwargs['target_mean'] = 0.2

    arrivals_all = get_or_generate_sequences(distribution, n, trials, seed, **kwargs)

    for error in error_levels:
        args_list = []
        for i in range(trials):
            arrivals = arrivals_all[i]
            predictions = add_prediction_error(arrivals, error)
            args_list.append((i, arrivals, predictions, lambda_values, rho))

        with ProcessPoolExecutor() as executor:
            trial_results = list(executor.map(compute_trial_data, args_list))

        classic_vals = [res["classic"] for res in trial_results]
        trust_vals = [res["trust"] for res in trial_results]
        ack_each_vals = [res["ack_each"] for res in trial_results]
        fixed_time_vals = [res["fixed_time"] for res in trial_results]

        classic_stats.append(np.mean(classic_vals))
        trust_stats.append(np.mean(trust_vals))
        ack_each_stats.append(np.mean(ack_each_vals))
        fixed_time_stats.append(np.mean(fixed_time_vals))

        for lam in lambda_values:
            vals = [res[lam] for res in trial_results]
            augmented_stats[lam].append(np.mean(vals))

    results_dict = {
        'classic': classic_stats,
        'trust': trust_stats,
        'ack_each': ack_each_stats,
        'fixed_time': fixed_time_stats,
    }
    results_dict.update(augmented_stats)
    save_results_to_csv(error_levels, results_dict, distribution, lambda_values)

    plt.figure(figsize=(10, 6))
    plt.plot(error_levels, classic_stats, 'k:', label='Classic (λ=1)')
    plt.plot(error_levels, trust_stats, 'r-.', label='Trust (λ=0)')
    plt.plot(error_levels, ack_each_stats, 'g--', label='ACK-after-each')
    plt.plot(error_levels, fixed_time_stats, 'b--', label='Fixed-time')
    for lam in lambda_values:
        plt.plot(error_levels, augmented_stats[lam], label=f'λ={lam}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Performance Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    output_file = os.path.join(RESULTS_DIR, f"cr_error_mean_comparison_{distribution}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches='tight')
    print(f"Plot saved: {output_file}")


if __name__ == "__main__":
    pcap_file = "202501011400.pcap"

    lambda_values = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    rho = 0.5
    n = 1000
    trials = 1000
    seed = 43

    mawi_seq_exists = check_sequence_exists('MAWI', n, trials, seed, 
                                             pcap_file=pcap_file, target_mean=0.2)

    if mawi_seq_exists:
        print("MAWI sequence file found, using cached data (no need for PCAP file)")
        plot_competitive_ratio_vs_errors(rho, lambda_values, distribution='MAWI', 
                                        timestamps=None, pcap_file=pcap_file,
                                        n=n, trials=trials, seed=seed)
    elif os.path.exists(pcap_file):
        print("MAWI sequence not cached, extracting from PCAP file")
        timestamps = get_or_extract_timestamps(pcap_file)
        plot_competitive_ratio_vs_errors(rho, lambda_values, distribution='MAWI', 
                                        timestamps=timestamps, pcap_file=pcap_file,
                                        n=n, trials=trials, seed=seed)
    else:
        print(f"Neither MAWI sequence nor PCAP file found. Skipping MAWI distribution.")

    plot_competitive_ratio_vs_errors(rho, lambda_values, distribution='Pareto', 
                                    alpha=3, scale=0.5, n=n, trials=trials, seed=seed)

    plot_competitive_ratio_vs_errors(rho, lambda_values, distribution='Poisson', 
                                    scale=0.3, n=n, trials=trials, seed=seed)

    plt.show()