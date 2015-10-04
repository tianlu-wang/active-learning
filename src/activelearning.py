__author__ = 'koala'

import os
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import re
import operator

class ActiveLearning(object):

    # named_entity_path = "./lib/wtextproc-2/named-entity"  # todo remember to pack the lib

    def __init__(self, iteration_times=1, init_training_num=1, increment=1, data_path=None, ):
        """
        initialize the activelearning class, classifier includes llama, svm and crf
        actually it's going to combine with other team and just add llama and svm;
        iteration_times is used to minimize the error;
        data_path_ltf temporarily don't know why
        :param iteration_times:
        :param init_training_num:
        :param increment:
        :param data_path:
        :return:
        """

        self.data_path = data_path
        ###### prepare two list
        raw_path = os.path.join('./', 'data_path_ltf')
        assert(os.path.exists(raw_path))
        f_raw = open(raw_path, 'r')
        self.raw_set = [line for line in f_raw.readlines()]
        gold_path = os.path.join('./', 'data_path_laf')
        assert(os.path.exists(gold_path))
        f_gold = open(gold_path, 'r')
        self.gold_set = [line for line in f_gold.readlines()]
        #########
        # print self.raw_set
        # print self.gold_set
        os.chdir(self.data_path)  # change the directory into the name tagger directory
        # os.system('pwd')
        self.init_training_num = init_training_num
        self.iteration_times = iteration_times
        self.increment = increment

        self.training_set = []
        self.test_set = []
        self.init_training_set = []
        self.incremental_training_set = []
        self.current_training_set = []
        self.rest_training_set = []

    def training_set_initialization(self):
        """
        randomly select several documents as the start point
        :return: index of the start point
        """
        print('===========================generating initial training set===============================')
        while len(self.init_training_set) < self.init_training_num:
            temp = random.randint(0, len(self.gold_set) - 1)
            self.init_training_set.append(temp if temp not in self.init_training_set else None)
        assert(index > 0 for index in self.init_training_set)
        print self.init_training_set  # this line is for debug
        return self.init_training_set

    def do_training(self, sampling_method):
        """
        add increment part in every loop
        :return:
        index: indexes of training files in every loop
        x[]: capacity of training set in every loop
        y_p[]: precision in every loop(write in file)
        y_r[]: recall in every loop(write in file)
        y_f[]: f1 score in every loop(write in file)
        """
        # select initial training set
        x = []
        index = []
        init_training_index = self.training_set_initialization()
        self.current_training_set = copy.deepcopy(init_training_index)
        # self.incremental_training_set = copy.deepcopy(init_training_index)  # todo why?
        ##############################################################
        MODEL_DIR='./new_data/model'      # directory for trained model
        LTF_DIR='./new_data/ltf'         # directory containing LTF files
        SYS_LAF_DIR='./new_data/output'   # directory for tagger output (LAF files)
        #TRAIN_SCP='./new_data/train.scp'  # script file containing paths to LAF files (one per line)
        #TEST_SCP='./new_data/test.scp'    # script file containing paths to LTF files (one per line)
        REF_LAF_DIR='./new_data/laf'      # directory containing gold standard LAF files
        PROBS_DIR = './new_data/probs'
        ################################################################
        for i in range(self.iteration_times):
            print('========================running iteration ' + str(i) + '========================')
            print('\tcurrent iteration training set size: '+str(len(self.current_training_set)))

            # =========================single iteration=====================
            train_list = ''
            for item in self.current_training_set:
                train_list += self.gold_set[item]+' '
            train_command = './train.py'+' '+MODEL_DIR+' '+LTF_DIR+' '+train_list  # todo: remember to delete the front path
            os.system(train_command)

            tag_list = ''
            temp = self.training_set
            for item in self.current_training_set:
                temp.remove(temp[item])
            for item in temp:
                tag_list += item+' '
            tag_command = './tagger.py'+' '+'-L'+' '+SYS_LAF_DIR+' '+' '+MODEL_DIR+' '+tag_list
            os.system(tag_command)

            score_command = './score.py'+' '+REF_LAF_DIR+' '+SYS_LAF_DIR+' '+LTF_DIR
            os.system(score_command)
            x.append(len(self.current_training_set))
            index.append(self.current_training_set)

            # =========================add new training samples========================
            self.rest_training_set = [d for d in self.training_set if self.training_set.index(d) not in self.current_training_set]

            # choose sampling method
            if sampling_method == 'uncertainty sampling':
                self.incremental_training_set = self.uncertainty_sampling().values()

            # elif sampling_method == 'random sampling':
            #     self.incremental_training_set = self.random_sampling()
            #
            # elif sampling_method == 'uncertainty k-means':
            #     self.incremental_training_set = self.uncertainty_k_means()
            # elif sampling_method == 'ne density':
            #     self.incremental_training_set = self.named_entity_density(i)
            # elif sampling_method == 'features based density':
            #     self.incremental_training_set = self.feature_based_density(i)

            self.current_training_set += self.incremental_training_set

        return x, index

    def uncertainty_sampling(self, probs_dir):
        print('\tgetting new training data...')

        entropy = dict()
        pattern = re.compile(r'(.*):(.*)')
        for root, dirs, files in os.walk(probs_dir):
            for file in files:
                sum_prob = 0
                f = open(probs_dir+'/'+file, 'r')
                i = 0
                for line in f.readlines():
                    m = re.match(pattern, line, flags=0)
                    if not m is None:
                        sum_prob += float(m.group(2))
                    i += 1
                mean_prob = sum_prob / float(i)  # take the mean of all tokens' probability as the entropy of a document
                entropy[file] = mean_prob

        sorted_entropy = sorted(entropy.items(), key=operator.itemgetter(1))

        training_set_to_add = []
        sample_size = self.increment
        if len(entropy) < self.increment:
            sample_size = len(entropy)
        for item in sorted_entropy[:sample_size]:
            sent_doc = item[0]
            training_set_to_add.append(self.raw_set.index(sent_doc.replace('_prob.txt', 'ltf.xml')))

        return training_set_to_add


def figure_plot(save_dir, learning_result):
    # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # determine major tick interval and minor tick interval
    x = learning_result.values()[0][0]
    y_acc = learning_result.values()[0][1]
    y_f = learning_result.values()[0][2]
    x_major_ticks = np.arange(min(x), max(x)+int(max(x)/10), int(max(x)/10))
    if max(y_acc) < min(y_f):
        y_major_ticks = np.arange(min(y_f), max(y_f)+0.05, 0.04)
    else:
        y_major_ticks = np.arange(min(y_f), max(y_acc)+0.05, 0.04)

    ax.set_xticks(x_major_ticks)
    # ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)

    # and a corresponding grid
    ax.grid(which='both')

    # or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=1)

    colors = ['r', 'b', 'g', 'k']
    markers = ['o', '*', 'v', '+']
    i = 0
    for s in learning_result.keys():
        if 0 not in learning_result[s][1]:
            acc_auc = metrics.auc(learning_result[s][0], learning_result[s][1])
            max_auc = metrics.auc(learning_result[s][0], [1]*len(learning_result[s][0]))
            plt.plot(learning_result[s][0], learning_result[s][1], colors[i]+markers[i]+'-',
                     label=s + ' acc (auc = %.2f/%d)'% (acc_auc, max_auc), ms=5, linewidth=2.0)
        if 0 not in learning_result[s][2]:
            f_auc = metrics.auc(learning_result[s][0], learning_result[s][2])
            max_auc = metrics.auc(learning_result[s][0], [1]*len(learning_result[s][0]))
            plt.plot(learning_result[s][0], learning_result[s][2], colors[i]+markers[i]+':',
                     label=s + ' f-score (auc = %.2f/%d)'% (f_auc, max_auc), ms=5, linewidth=2.0)
        i += 1

    plt.title('active learng and random sampling comparison')
    plt.legend(loc='lower right', prop={'size': 6})

    plt.margins(0.2)
    plt.xticks(rotation='vertical')  # make x ticks vertical
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(save_dir, 'comparison_curve.png'))
    plt.clf()

if __name__ == "__main__":
    act = ActiveLearning(data_path='/Users/koala/Documents/lab/Blender/LORELEI/active_learning/ne-tagger', init_training_num=5)
    # todo: must be absolute path '~'not work

    act.training_set_initialization()





