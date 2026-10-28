"""Review 2 K-means model using only Python built-ins.

Run from the repository root: python review2_kmeans_pure_python.py
The CSV contains synthetic judge scores. Its final column is held back until
evaluation; cluster fitting uses only the first three numeric columns.
"""

DATA_FILE = "data/review2_judge_scores.csv"
STATES = ("refusal", "partial_compliance", "full_violation")


def squared_distance(a, b):
    return sum((a[j] - b[j]) ** 2 for j in range(len(a)))


def load_data(path):
    features = []
    truth = []
    with open(path, encoding="utf-8") as stream:
        next(stream)  # column names
        for line in stream:
            fields = line.strip().split(",")
            if len(fields) != 4:
                raise ValueError("Expected three scores and one state per row")
            features.append([float(fields[j]) for j in range(3)])
            truth.append(STATES.index(fields[3]))
    return features, truth


def standardize(rows):
    count = len(rows)
    means = [sum(row[j] for row in rows) / count for j in range(3)]
    deviations = [
        (sum((row[j] - means[j]) ** 2 for row in rows) / count) ** 0.5
        for j in range(3)
    ]
    return [
        [(row[j] - means[j]) / deviations[j] for j in range(3)]
        for row in rows
    ]


def kmeans(rows, k=3, max_iterations=100):
    # Deterministic farthest-first initialization, using features only.
    centers = [rows[0][:]]
    while len(centers) < k:
        chosen = max(
            rows,
            key=lambda row: min(squared_distance(row, center) for center in centers),
        )
        centers.append(chosen[:])

    labels = [-1] * len(rows)
    for iteration in range(max_iterations):
        new_labels = [
            min(range(k), key=lambda j: squared_distance(row, centers[j]))
            for row in rows
        ]
        if new_labels == labels:
            break
        labels = new_labels
        groups = [[] for _ in range(k)]
        for row, label in zip(rows, labels):
            groups[label].append(row)
        for j in range(k):
            if groups[j]:
                centers[j] = [
                    sum(row[column] for row in groups[j]) / len(groups[j])
                    for column in range(3)
                ]
    inertia = sum(squared_distance(row, centers[label]) for row, label in zip(rows, labels))
    return labels, centers, inertia, iteration + 1


def matched_accuracy(labels, truth):
    # Labels are used here only after fitting, to score the clusters.
    mappings = (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    )
    return max(
        sum(mapping[label] == state for label, state in zip(labels, truth)) / len(truth)
        for mapping in mappings
    )


def main():
    raw_scores, truth = load_data(DATA_FILE)
    scores = standardize(raw_scores)
    labels, centers, inertia, iterations = kmeans(scores)
    print(f"Rows: {len(scores)}")
    print(f"Iterations: {iterations}")
    print(f"K-means distortion: {inertia:.3f}")
    print(f"Matched accuracy on synthetic states: {matched_accuracy(labels, truth):.3f}")
    print(f"Cluster sizes: {[labels.count(j) for j in range(3)]}")


if __name__ == "__main__":
    main()
