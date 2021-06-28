def demographic_parity(dataset, sensitive_columns, target_column):
    s1 = [row for row in dataset if row[sensitive_columns[0]] == 0 and row[sensitive_columns[1]] == 1]
    s2 = [row for row in dataset if row[sensitive_columns[0]] == 1 and row[sensitive_columns[1]] == 0]
    if (not len(s1)) or (not len(s2)):
        return 0
    positive_rate_s1 = [row[target_column] for row in s1].count(1) / float(len(s1))
    positive_rate_s2 = [row[target_column] for row in s2].count(1) / float(len(s2))
    return abs(positive_rate_s1 - positive_rate_s2)


def equalized_odds(dataset, predicted, sensitive_columns, target_column):
    true_positives = []
    false_positives = []
    for i in range(0, len(dataset)):
        if dataset[i][target_column] == predicted[i] and predicted[i] == 1:
            true_positives.append(dataset[i])
        elif dataset[i][target_column] != predicted[i] and predicted[i] == 1:
            false_positives.append(dataset[i])
    s1_true_positives = [row for row in true_positives if
                         row[sensitive_columns[0]] == 0 and row[sensitive_columns[1]] == 1]
    s2_true_positives = [row for row in true_positives if
                         row[sensitive_columns[0]] == 1 and row[sensitive_columns[1]] == 0]
    s1_false_positives = [row for row in false_positives if
                          row[sensitive_columns[0]] == 0 and row[sensitive_columns[1]] == 1]
    s2_false_positives = [row for row in false_positives if
                          row[sensitive_columns[0]] == 1 and row[sensitive_columns[1]] == 0]
    true_positive_rate_s1 = len(s1_true_positives) / float(len(true_positives))
    true_positive_rate_s2 = len(s2_true_positives) / float(len(true_positives))
    false_positive_rate_s1 = len(s1_false_positives) / float(len(false_positives))
    false_positive_rate_s2 = len(s2_false_positives) / float(len(false_positives))
    return {
        'true_positive_parity': abs(true_positive_rate_s1 - true_positive_rate_s2),
        'false_positive_parity': abs(false_positive_rate_s1 - false_positive_rate_s2)
    }


def demographic_parity_marital_status_sex(dataset, target_column):
    s1 = [row for row in dataset if row[5] == 1]  # urm
    s2 = [row for row in dataset if row[5] == 0]  # not urm
    if (not len(s1)) or (not len(s2)):
        return 0
    positive_rate_s1 = [row[target_column] for row in s1].count(1) / float(len(s1))
    positive_rate_s2 = [row[target_column] for row in s2].count(1) / float(len(s2))
    return abs(positive_rate_s1 - positive_rate_s2)


def equalized_odds_marital_status_sex(dataset, predicted, target_column):
    true_positives = []
    false_positives = []
    for i in range(0, len(dataset)):
        if dataset[i][target_column] == predicted[i] and predicted[i] == 1:
            true_positives.append(dataset[i])
        elif dataset[i][target_column] != predicted[i] and predicted[i] == 1:
            false_positives.append(dataset[i])
    s1_true_positives = [row for row in true_positives if row[5] == 1]
    s2_true_positives = [row for row in true_positives if row[5] == 0]
    s1_false_positives = [row for row in false_positives if row[5] == 1]
    s2_false_positives = [row for row in false_positives if row[5] == 0]
    true_positive_rate_s1 = len(s1_true_positives) / float(len(true_positives))
    true_positive_rate_s2 = len(s2_true_positives) / float(len(true_positives))
    false_positive_rate_s1 = len(s1_false_positives) / float(len(false_positives))
    false_positive_rate_s2 = len(s2_false_positives) / float(len(false_positives))
    return {
        'true_positive_parity': abs(true_positive_rate_s1 - true_positive_rate_s2),
        'false_positive_parity': abs(false_positive_rate_s1 - false_positive_rate_s2)
    }


def demographic_parity_marital_status_sex_v2(dataset, target_column):
    s1 = [row for row in dataset if row[5] == 1]  # urm
    s2 = [row for row in dataset if row[5] == 0]  # others
    if (not len(s1)) or (not len(s2)):
        return 0
    positive_rate_s1 = [row[target_column] for row in s1].count(1) / float(len(s1))
    positive_rate_s2 = [row[target_column] for row in s2].count(1) / float(len(s2))
    return abs(positive_rate_s1 - positive_rate_s2)


def equalized_odds_marital_status_sex_v2(dataset, predicted, target_column):
    true_positives = []
    false_positives = []
    for i in range(0, len(dataset)):
        if dataset[i][target_column] == predicted[i] and predicted[i] == 1:
            true_positives.append(dataset[i])
        elif dataset[i][target_column] != predicted[i] and predicted[i] == 1:
            false_positives.append(dataset[i])
    s1_true_positives = [row for row in true_positives if row[5] == 1]
    s2_true_positives = [row for row in true_positives if row[5] == 0]
    s1_false_positives = [row for row in false_positives if row[5] == 1]
    s2_false_positives = [row for row in false_positives if row[5] == 0]
    true_positive_rate_s1 = len(s1_true_positives) / float(len(true_positives))
    true_positive_rate_s2 = len(s2_true_positives) / float(len(true_positives))
    false_positive_rate_s1 = len(s1_false_positives) / float(len(false_positives))
    false_positive_rate_s2 = len(s2_false_positives) / float(len(false_positives))
    return {
        'true_positive_parity': abs(true_positive_rate_s1 - true_positive_rate_s2),
        'false_positive_parity': abs(false_positive_rate_s1 - false_positive_rate_s2)
    }