/**********************************************************************
 * Copyright (C) 2017 Christopher Morris <christopher.morris@udo.edu>
 *
 * This file is part of globalwl.
 *
 * globalwl can not be copied and/or distributed without the express
 * permission of Christopher Morris.
 *********************************************************************/

#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/WeisfeilerLehmanThreeLocal.h"
#include "src/WeisfeilerLehmanTwoLocal.h"
#include "src/WeisfeilerLehmanThreeGlobal.h"
#include "src/WeisfeilerLehmanTwoGlobal.h"
#include "src/GraphletKernel.h"
#include "src/ColorRefinementKernel.h"
#include "src/ShortestPathKernel.h"
using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Usage: ./glocalwc datasetname algorithmname";
        return 1;
    }

    string graph_database_name = argv[1];
    string kernel = argv[2];
    cout << graph_database_name << " " << kernel << endl;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(graph_database_name);

    GramMatrix gm;

    switch(kernel) {
    case "WL3L":
        WeisfeilerLehmanThreeLocal::WeisfeilerLehmanThreeLocal spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    case "ShortestPath":
        ShortestPathKernel::ShortestPathKernel spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    case "Graphlet":
        GraphletKernel::GraphletKernel gk(gdb);
        gm = gk.compute_gram_matrix(true);
        break;
    case "WL2L":
        WeisfeilerLehmanTwoLocal::WeisfeilerLehmanTwoLocal spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    case "WL3G":
        WeisfeilerLehmanThreeGlobal::WeisfeilerLehmanThreeGlobal spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    case "WL2G":
        WeisfeilerLehmanTwoGlobal::WeisfeilerLehmanTwoGlobal spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    case "ColorRefinement":
        ColorRefinement::ColorRefinementKernel spk(gdb);
        gm = spk.compute_gram_matrix(false);
        break;
    default:
        cout << "Argument not supported (Typo?)" << endl;
        break;
    }

    AuxiliaryMethods::write_gram_matrix(gm, graph_database_name + "_" + kernel);

    return 0;
}

