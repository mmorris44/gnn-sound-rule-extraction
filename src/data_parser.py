import random

import rdflib as rdf


def parse_as_nt(file):
    dataset = []
    graph = rdf.Graph()
    graph.parse(file, format='nt')
    for s, p, o in graph:
        dataset.append((str(s), str(p), str(o)))
    return dataset


def parse_as_tsv(file):
    inputfile = open(file, "r")
    lines = inputfile.readlines()
    dataset = []
    for line in lines:
        ent1, ent2, ent3 = line.split()
        dataset.append((ent1, ent2, ent3))
    return dataset


def parse_from_full_train_file(file, graph_target_split=0.1):
    # Use 10% for targets and other 90% for input, by default
    # Only handles binary predicates
    inputfile = open(file, "r")
    lines = inputfile.readlines()
    dataset = []
    predicates = set()
    for line in lines:
        ent1, ent2, ent3 = line.split()
        predicates.add(ent2)
        dataset.append((ent1, ent2, ent3))
    random.shuffle(dataset)
    split_index = int(len(dataset) * graph_target_split)
    targets = dataset[:split_index]
    input_graph = dataset[split_index:]
    predicate_list = list(predicates)
    predicate_list.sort()  # fixes the order as alphabetical
    return input_graph, targets, predicate_list


def parse(file):
    if file.endswith('.nt'):
        return parse_as_nt(file)
    elif file.endswith('.tsv') or file.endswith('.txt'):
        return parse_as_tsv(file)
    else:
        assert False, "Error, data format not supported. Use .nt or .tsv (or .txt, interpreted as a .tsv file)"
