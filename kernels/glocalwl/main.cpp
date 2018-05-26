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

    bool use_labels = false;
    int num_iterations = 1;
    bool use_iso_type = false;
    if (argc != 7) {
        cout << "Usage: ./glocalwc datasetname algorithmname -l/-nl num_iterations -i/-ni output_path\n";
        return 1;
    }

    if (argv[3] == "-l") {
        use_labels = true;
    }
    num_iterations = stoi(argv[4]);
    if (argv[5] == "-i") {
        use_iso_type = true;
    }


    string graph_database_name = argv[1];
    string kernel = argv[2];
    cout << graph_database_name << " " << kernel << endl;
    GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(graph_database_name);

    GramMatrix gm;

    if (kernel == "WL3L") {
        WeisfeilerLehmanThreeLocal::WeisfeilerLehmanThreeLocal w3l(gdb);
        gm = w3l.compute_gram_matrix(num_iterations, use_labels, use_iso_type);
    } else if (kernel == "ShortestPath") {
        ShortestPathKernel::ShortestPathKernel spk(gdb);
        gm = spk.compute_gram_matrix(use_labels);

    } else if (kernel == "Graphlet") {
        GraphletKernel::GraphletKernel gk(gdb);
        gm = gk.compute_gram_matrix(use_labels);

    } else if (kernel == "WL2L") {
        WeisfeilerLehmanTwoLocal::WeisfeilerLehmanTwoLocal w2l(gdb);
        gm = w2l.compute_gram_matrix(num_iterations, use_labels, use_iso_type);

    } else if (kernel == "WL3G") {
        WeisfeilerLehmanThreeGlobal::WeisfeilerLehmanThreeGlobal w3g(gdb);
        gm = w3g.compute_gram_matrix(num_iterations, use_labels, use_iso_type);

    } else if (kernel == "WL2G") {
        WeisfeilerLehmanTwoGlobal::WeisfeilerLehmanTwoGlobal w2g(gdb);
        gm = w2g.compute_gram_matrix(num_iterations, use_labels, use_iso_type);

    } else if (kernel == "ColorRefinement") {
        ColorRefinement::ColorRefinementKernel cr(gdb);
        gm = cr.compute_gram_matrix(num_iterations, use_labels);

    } else {
        cout << "Argument not supported (Typo?)" << endl;

    }
    AuxiliaryMethods::write_gram_matrix(gm, argv[6]);

    return 0;
}

