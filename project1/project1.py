import sys
import networkx as nx
import pandas as pd
import matplotlib
from collections import Counter
import numpy as np
import math
import scipy
import random
import copy

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def initcounts(bestguess,df,instances):
    counts = {}
    for node in bestguess:
        counts[node] = []
        parents = [pred for pred in bestguess.predecessors(node)]
        #print(parents)
        pcount = 0
        if len(parents) == 0:
            counts[node].append(np.zeros(instances[node]))
            for i in range(len(df)):
                counts[node][0][df.iloc[i][node]-1] += 1
            continue
        pdic = {}
        pcount = 0


        for i in range(len(df)):
            pinlist = []
            for parent in parents:
                pinlist.append(df.iloc[i][parent])
            pinstantiation = tuple(pinlist)
            #print(pinstantiation)

            if pinstantiation not in pdic:
                pdic[pinstantiation] = pcount
                pcount += 1
                counts[node].append(np.zeros(instances[node]))
                #print(counts[node])

            #print(pdic[pinstantiation])
            counts[node][pdic[pinstantiation]][df.iloc[i][node]-1] += 1

    return(counts)


def updatecounts(counts, bestguess,thenode,df,instances):
    x = [thenode]

    for node in x:
        counts[node] = []
        parents = [pred for pred in bestguess.predecessors(node)]
        pcount = 0
        if len(parents) == 0:
            counts[node].append(np.zeros(instances[node]))
            for i in range(len(df)):
                counts[node][0][df.iloc[i][node]-1] += 1
            continue
        pdic = {}
        pcount = 0


        for i in range(len(df)):
            pinlist = []
            for parent in parents:
                pinlist.append(df.iloc[i][parent])
            pinstantiation = tuple(pinlist)

            if pinstantiation not in pdic:
                pdic[pinstantiation] = pcount
                pcount += 1
                counts[node].append(np.zeros(instances[node]))

            counts[node][pdic[pinstantiation]][df.iloc[i][node]-1] += 1

    return(counts)


def bayesian_score(bestguess,counts,instances):
    score = 0
    for node in bestguess:
        npinstance = len(counts[node])
        for row in range(npinstance):
            score += (scipy.special.loggamma(instances[node]) - (scipy.special.loggamma(instances[node] + sum(counts[node][row]))))
            score += sum([scipy.special.loggamma((1+counts[node][row][r]) - scipy.special.loggamma(1)) for r in range(instances[node])])
    return(score)
    
def rand_neighbor(G):
    newgraph = copy.deepcopy(G)
    samplenodes = random.sample(list(G.nodes()), 2)

    if G.has_edge(samplenodes[0],samplenodes[1]):
        newgraph.remove_edge(samplenodes[0],samplenodes[1])
    else:
        newgraph.add_edge(samplenodes[0],samplenodes[1])
        
    return((newgraph, samplenodes[1]))


def localgraphsearch(G,df,instances):
    k_max = 10
    counts = initcounts(G,df,instances)
    currscore = bayesian_score(G, counts,instances)
    for i in range(k_max):
        print(i)
        tup = rand_neighbor(G)
        Gprime = tup[0]
        augnode = tup[1]
        #nx.draw(Gprime)
        if not(nx.is_directed_acyclic_graph(Gprime)):
            continue
        #print("passed the point of no return")
        countsprime = copy.deepcopy(counts)
        updatecounts(countsprime,Gprime,augnode,df,instances)
        nextscore = bayesian_score(Gprime,countsprime,instances)
        
        if nextscore > currscore:
            #print("I reign")
            print(nextscore)
            currscore = nextscore
            counts = copy.deepcopy(countsprime)
            G = copy.deepcopy(Gprime)

    return G


def compute(infile, outfile):
    df = pd.read_csv(infile)
    data = df.to_numpy()
    instances = {}
    for count, label in enumerate(df.columns):
        instances[label] = max(data[:,count])
        
    bestguess = nx.DiGraph()
    for label in df.columns:
        bestguess.add_node(label)
    
    answer = localgraphsearch(bestguess,df,instances)
    nx.readwrite.adjlist.write_adjlist(answer, outfile, "test.adjlist")


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
