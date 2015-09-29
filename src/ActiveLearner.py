__author__ = 'boliangzhang'

import os
import shutil
import operator
import matplotlib.pyplot as plt
import re
import cPickle
import math
import copy
import numpy as np
from subprocess import Popen, PIPE
from collections import OrderedDict
from sklearn import metrics
from sklearn.cluster import KMeans

#  from sklearn.metrics import pairwise_distances_argmin_min


class ActiveLearner(object):
    #intent_path = "~/Documents/lab/Blender/active-learning/Interactions/src/lib/wtextproc-2/intent"
    named_entity_path = "./lib/wtextproc-2/named-entity"  # todo remember to pack the lib



    def __init__(self, task, classifier, data=None, data_path=None):
        self.classifier = classifier
        self.task = task

        if data:
            self.data = data
        elif data_path:
            self.data = self.read_rovi_data(data_path)

        self.training_set = []
        self.test_set = []

        self.iteration_times = 20  # todo: maybe it needs changed
        self.increment = 0

        self.init_training_set = []
        self.incremental_training_set = []
        self.current_training_set = []
        self.rest_training_set = []

        if self.task == 'intent_tagger':
            os.chdir(self.intent_path)   # change dir to project root directory
        elif self.task == 'named_entity_tagger':
            os.chdir(self.named_entity_path)  # change dir to project root directory
        else:
            print('task directory incorrect')

    def read_rovi_data(self, path):
        f = open(path).read()

        if self.task == 'intent_tagger':
            data = f.splitlines()
        elif self.task == 'named_entity_tagger':
            data = f.strip().split('\n\n')
        else:
            print('task incorrect when reading rovi data')

        return data[:1000]  # todo:don't why it's 1000

    def training_set_initialization(self):
        print('generating initial training set...')
        # === generate features for initial training set === #
        # save training set to train file
        f = open('data/train', 'w')
        if self.task == 'named_entity_tagger':
            f.write('\n\n'.join(self.training_set))
            f.close()
        elif self.task == 'intent_tagger':
            f.write('\n'.join(self.training_set))
            f.close()

        p = Popen(['wnltp', '--input', 'data/train.desc', '-p', 'train.pipe',
                   '-d', 'datfile: train.dat', '-d', 'lblfile: train.lbl'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()

        f = open('train.dat')
        if self.task == 'named_entity_tagger':
            raw_features = f.read().strip().split('\n\n')
            raw_features = [f.splitlines() for f in raw_features]
        elif self.task == 'intent_tagger':
            raw_features = f.read().strip().split('\n')

        p = Popen(['laclstrain', '--stem=train', '--parm=llama.parm', '--output=llama.mdl'], stdout=PIPE, stderr=PIPE)

        p.communicate()

        # load feature bag
        p = Popen(['ladictc', '-p', 'llama.mdl/dict.cdb'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        feature_bag = [' '.join(line.split()[1:-1]) for line in stdout.splitlines()]

        binary_features = []

        if self.task == 'named_entity_tagger':
            for sent_feature in raw_features:
                binary_sent_f = []
                for token_f in sent_feature:
                    b_features = []
                    for item in feature_bag:
                        if item in token_f:
                            b_features.append(1)
                        else:
                            b_features.append(0)
                    binary_sent_f.append(b_features)
                # sentence feature is the sum of token features
                binary_sent_f = np.array(binary_sent_f)
                binary_sent_f = [sum(binary_sent_f[:, i]) for i in xrange(len(binary_sent_f[0]))]
                # normalize sentence feature
                binary_sent_f = binary_sent_f / np.linalg.norm(binary_sent_f)

                binary_features.append(binary_sent_f)
        elif self.task == 'intent_tagger':
            for sent_feature in raw_features:
                binary_sent_f = []
                for item in feature_bag:
                    if item in sent_feature:
                        binary_sent_f.append(1)
                    else:
                        binary_sent_f.append(0)
                binary_features.append(binary_sent_f)

        initial_training_index = []

        # =========== K means clustering =========== #
        # count labels
        labels = set()
        if self.task == 'named_entity_tagger':
            for line in self.training_set:
                line = line.splitlines()
                for l in line:
                    l = l.split()
                    if l[1].startswith('B'):
                        labels.add(l[1])
        elif self.task == 'intent_tagger':
            for line in self.training_set:
                labels.add(line.split('A')[0])
        k = len(labels)
        # k = 10
        # k = self.increment

        k_means = KMeans(n_clusters=k)

        transformed_training_set = k_means.fit_transform(binary_features)

        j = 0
        while True:
            if len(initial_training_index) > self.increment:
                break
            for i in xrange(k):
                ind = np.argsort(transformed_training_set[:, i])[j]
                initial_training_index.append(ind)
            j += 1

        # =========== density based selecting ============ #
        # feature_sim = []
        # for sent1 in binary_features:
        #     sim = []
        #     for sent2 in binary_features:
        #         # compute feature vector similarity
        #         sim.append(self.vector_sim(sent1, sent2))
        #     feature_sim.append(sim)
        #
        # # compute density of sentences
        # feature_density = []
        # for i in xrange(len(feature_sim)):
        #     # compute density
        #     d = sum([feature_sim[i][j] for j in xrange(len(feature_sim[i])) if j != i]) / float(len(feature_sim)-1)
        #     feature_density.append(d)
        #
        # feature_density = [sent_d / max(feature_density) for sent_d in feature_density]  # normalization
        #
        # scores = dict()
        # for i in xrange(len(feature_density)):
        #     scores[i] = feature_density[i]
        #
        # sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        #
        # for item in sorted_scores[:self.increment]:
        #     sent_id = item[0]
        #     initial_training_index.append(sent_id)

        # =============== generate initial training set =================== #

        initial_training_index = initial_training_index[:self.increment]
        assert len(initial_training_index) == self.increment

        initial_training_set = [self.training_set[i] for i in initial_training_index]

        print('Done')

        return initial_training_set

    def do_training(self, sampling_method):
        x = []
        y_acc = []
        y_f = []

        # export test set to file for classifier
        f = open('data/test', 'w')
        if self.task == 'intent_tagger':
            f.write('\n'.join(self.test_set))
            f.close()
        elif self.task == 'named_entity_tagger':
            f.write('\n\n'.join(self.test_set))
            f.close()
        print('\ttest data size: ' + str(len(self.test_set)))

        # select initial training set
        self.current_training_set = copy.deepcopy(self.init_training_set)
        self.incremental_training_set = copy.deepcopy(self.init_training_set)

        if os.path.isfile('train.dat'):
            os.remove('train.dat')
        if os.path.isfile('train.lbl'):
            os.remove('train.lbl')

        for i in xrange(50):
            print('== running iteration '+str(i)+' (' + sampling_method + ')...')
            print('\tcurrent iteration training set size: '+str(len(self.current_training_set)))
            # generate single iteration training file
            if self.task == 'intent_tagger':
                f = open('data/train', 'w')
                f.write('\n'.join(self.incremental_training_set))
                f.close()
            elif self.task == 'named_entity_tagger':
                f = open('data/train', 'w')
                f.write('\n\n'.join(self.incremental_training_set))
                f.close()

            # ========================== single iteration ============================ #
            # generate features from training data
            print('\trunning feature generating...'),
            p = Popen(['wnltp', '--input', 'data/train.desc', '-p', 'train.pipe',
                       '-d', 'datfile: incremental_train.dat', '-d', 'lblfile: incremental_train.lbl'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            print('Done')

            # incremental add training instances to improve speed
            train_dat = open('train.dat', 'a')
            train_dat.write(open('incremental_train.dat').read())
            train_dat.close()
            train_lbl = open('train.lbl', 'a')
            train_lbl.write(open('incremental_train.lbl').read())
            train_lbl.close()

            # train a new model on the newly generated features using laclstrain
            print('\trunning llama...'),
            if os.path.isdir('llama.mdl'):
                shutil.rmtree('llama.mdl')
            if self.classifier == 'llama':
                p = Popen(['laclstrain', '--stem=train', '--parm=llama.parm', '--output=llama.mdl'], stdout=PIPE, stderr=PIPE)
            if self.classifier == 'svm':
                p = Popen(['laclstrain', '--stem=train', '--parm=svm.parm', '--output=svm.mdl'], stdout=PIPE, stderr=PIPE)
            if self.classifier == 'crf':
                p = Popen(['/home/boliangzhang/Documents/nlp-next/model/nl/ML/crf2/bin/crftrain',
                       '--stem=train', '--target=ne', '--model=crf.mdl'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            print('\tDone')

            # generate test result
            print('\tgenerating test results...'),
            if self.classifier == 'llama':
                p = Popen(['wnltp', '--input', 'data/test.desc', '-p', 'eval.llama.pipe'], stdout=PIPE, stderr=PIPE)
            elif self.classifier == 'svm':
                p = Popen(['wnltp', '--input', 'data/test.desc', '-p', 'eval.svm.pipe'], stdout=PIPE, stderr=PIPE)
            elif self.classifier == 'crf':
                p = Popen(['wnltp', '--input', 'data/test.desc', '-p', 'eval.crf.pipe'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            f = open('./predict_output_crnt_itr', 'w')
            # print(stderr)
            f.write(stdout)
            f.close()
            print('Done')

            # evaluate test result
            print('\tevaluating test result...'),
            if self.task == 'named_entity_tagger':
                p = Popen(['./eval-nebrk.perl', 'predict_output_crnt_itr'], stdout=PIPE, stderr=PIPE)
            elif self.task == 'intent_tagger':
                p = Popen(['./eval-nltp.perl', 'predict_output_crnt_itr'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            print(stderr),
            print('Done')

            out_path = os.path.join('itr_eval_rslt', 'eval_result_itr'+str(i))
            f_out = open(out_path, 'w')
            f_out.write(stdout)
            f_out.close()

            if self.task == 'named_entity_tagger':
                # when use eval-nebrk.perl
                accuracy = 0
                f_score = float(stdout.strip().split('\n')[-1].split()[-3])
            elif self.task == 'intent_tagger':
                accuracy = float(stdout.splitlines()[-2].split(':')[-1])
                f_score = 0

            print('\tcurrent iteration accuracy: ' + str(accuracy))
            if f_score:
                print('\tcurrent iteration f score: ' + str(f_score))

            x.append(len(self.current_training_set))
            y_acc.append(accuracy)
            y_f.append(f_score)

            if len(self.current_training_set)+self.increment > len(self.training_set):
                break

            # ===================== add new training instances ======================== #
            self.rest_training_set = [d for d in self.training_set if d not in self.current_training_set]

            # choose sampling method
            if sampling_method == 'uncertainty sampling':
                self.incremental_training_set = self.uncertainty_sampling().values()

            elif sampling_method == 'random sampling':
                self.incremental_training_set = self.random_sampling()

            elif sampling_method == 'uncertainty k-means':
                self.incremental_training_set = self.uncertainty_k_means()
            elif sampling_method == 'ne density':
                self.incremental_training_set = self.named_entity_density(i)
            elif sampling_method == 'features based density':
                self.incremental_training_set = self.feature_based_density(i)

            self.current_training_set += self.incremental_training_set

        return x, y_acc, y_f

    # ================ sampling methods ================= #

    def uncertainty_sampling(self, sample_size=None):
        print('\tgetting new training data...'),
        # generate active learning test set
        f = open('data/active_test', 'w')
        if self.task == 'named_entity_tagger':
            f.write('\n\n'.join(self.rest_training_set))
        elif self.task == 'intent_tagger':
            f.write('\n'.join(self.rest_training_set))
        f.close()
        print('Done')

        # generate test result
        print('\tgenerating test results...'),
        if self.classifier == 'llama':
            p = Popen(['wnltp', '-vvv', '--input', 'data/active_test.desc', '-p', 'eval.llama.pipe'], stdout=PIPE, stderr=PIPE)
        elif self.classifier == 'svm':
            p = Popen(['wnltp', '-vvv', '--input', 'data/active_test.desc', '-p', 'eval.svm.pipe'], stdout=PIPE, stderr=PIPE)
        elif self.classifier == 'crf':
            p = Popen(['wnltp', '-vvv', '--input', 'data/active_test.desc', '-p', 'eval.crf.pipe'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        f = open('./predict_output_crnt_itr', 'w')
        f.write(stdout)
        f.close()
        f = open('./verbose_output', 'w')
        f.write(stderr)
        f.close()
        print('Done')

        # for each item to tag, possibilities of all tags will be assigned and tag with highest possibility is chosen.
        if self.task == 'named_entity_tagger':
            tagging_result = self.load_tagging_result()
        elif self.task == 'intent_tagger':
            tagging_result = self.load_intent_tagging_result()

        entropy = dict()
        for sent_id in tagging_result.keys():
            if self.task == 'named_entity_tagger':
                e_list = []
                for token in tagging_result[sent_id]:
                    e = self.compute_entropy(token)
                    e_list.append(e)
                # here to determine sentence entropy by its tokens, various method applied
                if e_list:
                    # ----------- mean of all token entropy ----------- #
                    # entropy[sent_id] = sum(e_list) / len(e_list)

                    # ----------- maximum of all token entropy ----------- #
                    # entropy[sent_id] = max(e_list)

                    # ----------- minimum of all token entropy ------------ #
                    # entropy[sent_id] = min(e_list)

                    # ----------- maximum margin of all tokens (max - min) ----------- #
                    entropy[sent_id] = max(e_list) - min(e_list)  # currently the best for uncertainty sampling
                else:
                    entropy[sent_id] = 0

            elif self.task == 'intent_tagger':
                e = self.compute_entropy(tagging_result[sent_id].values())
                entropy[sent_id] = e

        sorted_entropy = sorted(entropy.items(), key=operator.itemgetter(1), reverse=True)

        training_set_to_add = dict()
        if not sample_size:
            sample_size = self.increment
        for item in sorted_entropy[:sample_size]:
            sent_id = item[0]
            training_set_to_add[sent_id] = self.rest_training_set[sent_id]

        return training_set_to_add

    def feature_based_density(self, iteration):
        sample_size = self.increment * 2
        uncertainty_samples = self.uncertainty_sampling(sample_size)

        if len(uncertainty_samples) < self.increment:
            return uncertainty_samples

        # get feature for each token in sentences
        if self.task == 'named_entity_tagger':
            features = self.load_ne_features()
        elif self.task == 'intent_tagger':
            features = self.load_intent_features()

        # vector similarity sum token features in column direction
        for sent_id in features:
            features[sent_id] = [sum(features[sent_id][:, i]) for i in xrange(len(features[sent_id][0]))]
            features[sent_id] = features[sent_id] / np.linalg.norm(features[sent_id])

        # compute sentence similarities which will be used to compute density
        sent_sim = []
        for sent1 in features.values():
            sim = []
            for sent2 in features.values():
                if self.task == 'named_entity_tagger':
                    # compute feature vector similarity
                    # sim.append(self.dtw(sent1, sent2))
                    sim.append(self.vector_sim(sent1, sent2))
                elif self.task == 'intent_tagger':
                    sim.append(self.vector_sim(sent1, sent2))
            sent_sim.append(sim)

        # compute mean similarity
        mean_sim = sum([sum(item) for item in sent_sim]) / len(sent_sim)**2

        # compute density of sentences
        sent_density = []
        for i in xrange(len(sent_sim)):
            # compute density
            d = sum([sent_sim[i][j] for j in xrange(len(sent_sim[i])) if j != i]) / float(len(sent_sim)-1)
            sent_density.append(d)

        sent_density = [sent_d / max(sent_density) for sent_d in sent_density]  # normalization

        # compute informativeness of name entity based on entropy measure
        # get entropy for the named entity
        if self.task == 'named_entity_tagger':
            tagging_result = self.load_tagging_result()
        elif self.task == 'intent_tagger':
            tagging_result = self.load_intent_tagging_result()

        entropy = OrderedDict()
        for sent_id in tagging_result.keys():
            if self.task == 'named_entity_tagger':
                e_list = []
                for token in tagging_result[sent_id]:
                    e = self.compute_entropy(token)
                    e_list.append(e)
                # here to determine sentence entropy by its tokens, various method applied
                if e_list:
                    entropy[sent_id] = max(e_list) - min(e_list)  # currently the best for uncertainty sampling
                else:
                    entropy[sent_id] = 0

            elif self.task == 'intent_tagger':
                e = self.compute_entropy(tagging_result[sent_id].values())
                entropy[sent_id] = e

        sent_info = entropy.values()

        if max(sent_info) != 0:
            sent_info = [sent_f/max(sent_info) for sent_f in sent_info]  # normalization

        scores = dict()
        for i in xrange(len(sent_info)):
            # parameter = iteration / (len(self.training_set)/self.increment) * 1.0
            if iteration < 3:
                parameter = 0
            else:
                parameter = 1
            # parameter = 0.6
            s = parameter * sent_info[i] + (1 - parameter) * sent_density[i]
            scores[i] = s

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        training_set_to_add = []
        added_sent_id = []
        # for item in sorted_scores[:self.increment]:
        #     sent_id = item[0]
        #     if sent_id not in added_sent_id:
        #         training_set_to_add.append(self.rest_training_set[sent_id])
        #         added_sent_id.append(sent_id)

        # add constraint where instance is ignored if it's similar to instances in current batch. This contraints
        # tries to select less similar instances to add to new training data.
        for item in sorted_scores:
            if len(added_sent_id) == self.increment:
                break
            elif len(added_sent_id) > 0:
                sent_id = item[0]
                repeat_flag = 0
                for s_id in added_sent_id:
                    if sent_sim[sent_id][s_id] > mean_sim:
                        repeat_flag = 1
                        break
                if repeat_flag:
                    continue
            else:
                sent_id = item[0]

            training_set_to_add.append(self.rest_training_set[sent_id])
            added_sent_id.append(sent_id)

        if len(training_set_to_add) < self.increment:
            for item in sorted_scores:
                if len(training_set_to_add) == self.increment:
                    break
                sent_id = item[0]
                training_set_to_add.append(self.rest_training_set[sent_id])

        return training_set_to_add

    def named_entity_density(self, iteration):
        sample_size = self.increment * 2
        uncertainty_samples = self.uncertainty_sampling(sample_size)

        if len(uncertainty_samples) <= self.increment:
            return uncertainty_samples.values()

        # for each item to tag, possibilities of all tags will be assigned and tag with highest possibility is chosen.
        if self.task == 'named_entity_tagger':
            features = self.load_ne_features()
        elif self.task == 'intent_tagger':
            features = self.load_intent_features()

        # load predict results
        predicted_result = OrderedDict()  # {sent_id: {token: predicted_result}}
        sent_id = 0
        predict_output_crnt_itr = open('predict_output_crnt_itr').read().strip().split('\n\n')
        for sent in predict_output_crnt_itr:
            sent_result = OrderedDict()
            index = 0
            for line in sent.split('\n'):
                sent_result[index] = line.split()[2]
                index += 1
            predicted_result[sent_id] = sent_result
            sent_id += 1

        # compute density
        class NE(object):
            def __init__(self, token_features, index, sent_id):
                self.token_features = token_features
                self.index = index
                self.sent_id = sent_id
                self.density = None
                self.length = 0

        # find named entities
        named_entities = []
        for sent_id in predicted_result.keys():
            sent_result = predicted_result[sent_id].values()
            assert len(sent_result) == len(features[sent_id])
            ne_length = 0
            for i in xrange(len(sent_result)+1):
                if (i == len(sent_result) or sent_result[i] == 'O') and ne_length:
                    f = features[sent_id][i-ne_length:i]

                    # vector similarity sum token features in column direction
                    f = [sum(f[:, k]) for k in xrange(len(f[0]))]
                    f /= np.linalg.norm(f)

                    index = i-ne_length
                    ne = NE(f, index, sent_id)
                    ne.length = ne_length
                    named_entities.append(ne)
                    ne_length = 0
                elif i == len(sent_result) or sent_result[i] == 'O':
                    continue
                else:
                    ne_length += 1

        # compute ne similarities which will be used to compute density
        ne_sim = []
        for ne1 in named_entities:
            sim = []
            for ne2 in named_entities:
                # compute feature vector similarity
                # sim.append(self.dtw(ne1.token_features, ne2.token_features))
                sim.append(self.vector_sim(ne1.token_features, ne2.token_features))
            ne_sim.append(sim)

        # compute mean similarity
        mean_sim = sum([sum(item) for item in ne_sim]) / len(ne_sim)**2

        # compute density of named entities
        ne_density = []
        for i in xrange(len(named_entities)):
            # compute density
            d = sum([ne_sim[i][j] for j in xrange(len(ne_sim[i])) if j != i]) / float(len(named_entities)-1)
            ne_density.append(d)

        ne_density = [ne_d / max(ne_density) for ne_d in ne_density]  # normalization

        # compute informativeness of name entity based on entropy measure
        ne_info = []
        # get entropy for the named entity
        if self.task == 'named_entity_tagger':
            tagging_result = self.load_tagging_result()
        elif self.task == 'intent_tagger':
            tagging_result = self.load_intent_tagging_result()

        entropy = dict()
        for sent_id in tagging_result.keys():
            if self.task == 'named_entity_tagger':
                e_list = []
                for token in tagging_result[sent_id]:
                    e = self.compute_entropy(token)
                    e_list.append(e)
                # here to determine sentence entropy by its tokens, various method applied
                if e_list:
                    entropy[sent_id] = e_list  # currently the best for uncertainty sampling
                else:
                    entropy[sent_id] = [0]
        for ne in named_entities:
            ne_info.append(sum(entropy[ne.sent_id][ne.index:ne.index+ne.length]) / float(ne.length))

        ne_info = [ne_f/max(ne_info) for ne_f in ne_info]  # normalization

        scores = dict()
        for i in xrange(len(named_entities)):
            # parameter = iteration**3 / (len(self.training_set)/self.increment)**3.0
            if iteration < 3:
                parameter = 0
            else:
                parameter = 1
            # parameter = 0.6
            s = parameter * ne_info[i] + (1 - parameter) * ne_density[i]
            scores[i] = s

        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        training_set_to_add = []
        added_sent_id = []
        added_ne_index = []
        # for item in sorted_scores[:self.increment]:
        #     sent_id = named_entities[item[0]].sent_id
        #     if sent_id not in added_sent_id:
        #         training_set_to_add.append(self.rest_training_set[sent_id])
        #         added_sent_id.append(sent_id)

        # add constraint where instance is ignored if it's similar to instances in current batch. This contraints
        # tries to select less similar instances to add to new training data.
        for item in sorted_scores:
            if len(training_set_to_add) == self.increment:
                break
            elif len(training_set_to_add) > 0:
                sent_id = named_entities[item[0]].sent_id
                if sent_id in added_sent_id:
                    continue
                repeat_flag = 0
                for n_i in added_ne_index:
                    if ne_sim[item[0]][n_i] > mean_sim:
                        repeat_flag = 1
                        break
                if repeat_flag:
                    continue
            else:
                sent_id = named_entities[item[0]].sent_id

            training_set_to_add.append(self.rest_training_set[sent_id])
            added_sent_id.append(sent_id)
            added_ne_index.append(item[0])

        for item in sorted_scores[:self.increment]:
            if len(training_set_to_add) < self.increment:
                sent_id = named_entities[item[0]].sent_id
                if sent_id not in added_sent_id:
                    training_set_to_add.append(self.rest_training_set[sent_id])
                    added_sent_id.append(sent_id)

        # when added sentences are less than increment, add sentences using uncertainty criteria
        if len(training_set_to_add) < self.increment:
            # the rest training instances are added based on uncertainty sampling
            entropy = dict()
            for sent_id in tagging_result.keys():
                if self.task == 'named_entity_tagger':
                    e_list = []
                    for token in tagging_result[sent_id]:
                        e = self.compute_entropy(token)
                        e_list.append(e)
                    # here to determine sentence entropy by its tokens, various method applied
                    if e_list:
                        # ----------- mean of all token entropy ----------- #
                        # entropy[sent_id] = sum(e_list) / len(e_list)

                        # ----------- maximum of all token entropy ----------- #
                        # entropy[sent_id] = max(e_list)

                        # ----------- minimum of all token entropy ------------ #
                        # entropy[sent_id] = min(e_list)

                        # ----------- maximum margin of all tokens (max - min) ----------- #
                        entropy[sent_id] = max(e_list) - min(e_list)  # currently the best for uncertainty sampling
                    else:
                        entropy[sent_id] = 0

            sorted_entropy = sorted(entropy.items(), key=operator.itemgetter(1), reverse=True)

            for item in sorted_entropy:
                sent_id = item[0]
                if len(training_set_to_add) == self.increment:
                    break
                if sent_id not in added_sent_id:
                    training_set_to_add.append(self.rest_training_set[sent_id])

            assert len(training_set_to_add) == self.increment

            # training_set_to_add = random.shuffle(self.rest_training_set)[:self.increment]

            return training_set_to_add

        assert len(training_set_to_add) == self.increment

        return training_set_to_add

    def random_sampling(self):
        return self.rest_training_set[:self.increment]

    # ===================== utilities ================== #

    def load_ne_features(self):
        # load feature bag
        p = Popen(['ladictc', '-p', 'llama.mdl/dict.cdb'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        feature_bag = [' '.join(line.split()[1:-1]) for line in stdout.splitlines()]

        verbose_output = open('verbose_output').read()
        viterbi_beg = [beg.start() for beg in re.finditer("--> LlamaViterbi::processToken \[<EOS>\]", verbose_output)]
        viterbi_end = [end.start() for end in re.finditer("<-- LlamaViterbi::processToken \[<EOS>\]", verbose_output)]
        assert len(viterbi_beg) == len(viterbi_end)

        sent_feature = OrderedDict()
        for i in range(1, len(viterbi_beg), 2):
            viterbi_output = verbose_output[viterbi_beg[i]:viterbi_end[i]]
            features = []
            viterbi_output = viterbi_output.split('---------')
            first_token = True
            for token_output in viterbi_output:
                for line in token_output.splitlines():
                    if 'feature input:' in line:
                        f = line.split('feature input:')[1].strip()
                        f_vector = []
                        for item in feature_bag:
                            if item in f:
                                f_vector.append(1)
                            else:
                                f_vector.append(0)
                        # norm feature vector
                        f_vector = f_vector / np.linalg.norm(f_vector)
                        if first_token is True:
                            features.append(f_vector)
                            first_token = False
                            continue
                        features.append(f_vector)
                        break
            sent_feature[i/2] = np.array(features)

        return sent_feature

    def load_intent_features(self):
        # load feature bag
        p = Popen(['ladictc', '-p', 'llama.mdl/dict.cdb'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        feature_bag = [' '.join(line.split()[1:-1]) for line in stdout.splitlines()]

        f = open('verbose_output').read().splitlines()
        sent_feature = dict()
        sent_id = 0
        for i in xrange(len(f)):
            if '--> LlamaClassify::processToken [<EOS>]' in f[i]:
                intent_feature = f[i-2]
                if 'feature input' not in intent_feature:
                    intent_feature = f[i-3]
                word_feature = intent_feature.split(':')[1]
                f_vector = []
                for item in feature_bag:
                    if item in word_feature:
                        f_vector.append(1)
                    else:
                        f_vector.append(0)
                # norm feature vector
                f_vector = f_vector / np.linalg.norm(f_vector)

                sent_feature[sent_id] = f_vector
                sent_id += 1

        return sent_feature

    # compute named entity similarity by dynamic time warping
    def dtw(self, x, y):
        dtw_array = [[0]*(len(y)+1) for i in xrange(len(x)+1)]
        for i in xrange(len(x)):
            dtw_array[i][0] = 0
        for i in xrange(len(y)):
            dtw_array[0][i] = 0
        dtw_array[0][0] = 0
        for i in xrange(1, len(x)+1):
            for j in xrange(1, len(y)+1):
                v_1 = x[i-1]
                v_2 = y[j-1]
                simi = np.dot(v_1, v_2)
                dtw_array[i][j] = simi + max(dtw_array[i-1][j],
                                             dtw_array[i][j-1],
                                             dtw_array[i-1][j-1])
        return dtw_array[len(x)][len(y)]

    def vector_sim(self, x, y):
        return float(np.dot(x, y))

    def load_intent_tagging_result(self):
        ############################## named entity tagger ############################33
        # parse_wnltp_verbose_output():
        f = open('verbose_output').read().splitlines()
        sent_weight = dict()
        sent_id = 0
        for i in xrange(len(f)):
            if '--> LlamaClassify::processToken [<EOS>]' in f[i]:
                intent_weight = f[i-1]
                if 'olabs' not in intent_weight:
                    intent_weight = f[i-2]
                weights = dict()
                w = intent_weight.split('olabs:')[1].strip()
                w = re.findall("'([^']*)'", w)
                for k in range(0, len(w), 2):
                    if float(w[k+1]) == 0:  # smooth 0
                        w[k+1] = 0.0001
                    weights[w[k]] = float(w[k+1])
                sent_weight[sent_id] = weights
                sent_id += 1

        # normalization
        for sent_id in sent_weight:ug
            s = sum([float(v) for v in sent_weight[sent_id].values()])
            for intent_label in sent_weight[sent_id].keys():
                sent_weight[sent_id][intent_label] = float(sent_weight[sent_id][intent_label]) / s

        return sent_weight

    # return tagging reuslt by sentence.
    # for ne tagger: {sentence: [[p(tag1), p(tag2),...], ...]}
    # for intent tagger: {sentence: [p(tag1), p(tag2),...]}
    # 'sentence' is one training instance in training set.
    def load_tagging_result(self):
        ############################## named entity tagger ############################33
        # parse_wnltp_verbose_output():
        f = open('verbose_output').read()
        viterbi_beg = [beg.start() for beg in re.finditer("--> LlamaViterbi::processToken \[<EOS>\]", f)]
        viterbi_end = [end.start() for end in re.finditer("<-- LlamaViterbi::processToken \[<EOS>\]", f)]
        assert len(viterbi_beg) == len(viterbi_end)

        sent_weight = dict()
        for i in range(1, len(viterbi_beg), 2):
            viterbi_output = f[viterbi_beg[i]:viterbi_end[i]]
            weights = []
            viterbi_output = viterbi_output.split('---------')
            first_token = True
            for token_output in viterbi_output:
                token_w = []
                for line in token_output.splitlines():
                    if 'olabs:' in line:
                        w = line.split('olabs:')[1].strip()
                        w = re.findall("'([^']*)'", w)
                        if first_token is True:
                            weights.append([w])
                            first_token = False
                            continue
                        token_w.append(w)
                if token_w:
                    weights.append(token_w)

            # compute average weight for each token
            sent_d = dict()
            for j in xrange(0, len(weights)):
                token_w = weights[j]
                token_d = dict()
                for w in token_w:
                    for k in range(0, len(w), 2):
                        if float(w[k+1]) == 0:  # smooth 0
                            w[k+1] = 0.0001
                        try:
                            token_d[w[k]].append(float(w[k+1]))
                        except KeyError:
                            token_d[w[k]] = [float(w[k+1])]
                sent_d[j] = token_d

            s_weight = dict()
            for sent_id in sent_d.keys():
                token_w = dict()
                for e_type in sent_d[sent_id]:
                    e_type_weights = sent_d[sent_id][e_type]
                    token_w[e_type] = sum(e_type_weights) / float(len(e_type_weights))
                s_weight[sent_id] = token_w

            sent_weight[(i-1)/2] = s_weight

        tags = []
        for sent_id in sent_weight.keys():
            sent_w = sent_weight[sent_id]
            for token_w in sent_w.values():
                tags += token_w.keys()
        tags = list(set(tags))

        # normalization
        for sent_id in sent_weight:
            for token_id in sent_weight[sent_id]:
                token_w = sent_weight[sent_id][token_id]
                # normalize
                # s = sum([1/(1+math.exp(-1*v)) for v in token_w.values()])
                s = sum([1/float(v) for v in token_w.values()])
                for e_type in token_w.keys():
                    # token_w[e_type] = 1/(1+math.exp(-1*token_w[e_type]))/s
                    token_w[e_type] = 1/float(token_w[e_type])/s

        result = dict()
        for sent_id in sent_weight:
            probablities = []
            for token_id in sent_weight[sent_id]:
                token_w = sent_weight[sent_id][token_id]
                p = []
                for t in tags:
                    # if t == 'O':
                    #     p.append(0)
                    #     continue
                    try:
                        p.append(token_w[t])
                    except KeyError:
                        p.append(0)

                # disregard tokens labeled 'O'
                # sorted_token_w = sorted(token_w.items(), key=operator.itemgetter(1), reverse=True)
                # if sorted_token_w[0][0] == "O":
                #     continue

                probablities.append(p)

            result[sent_id] = probablities

        return result

    def compute_entropy(self, l):
        return sum([i*math.log(i) for i in l if i != 0])*(-1)

    def data_distribution_analysis(self):
        # training data and test data analysis
        labels = dict()
        for item in self.training_set:
            if self.task == 'named_entity_tagger':
                for line in item.splitlines():
                    try:
                        labels[line.split(' ')[1]] += 1
                    except KeyError:
                        labels[line.split(' ')[1]] = 1
            elif self.task == 'intent_tagger':
                try:
                    labels[item.split('A')[0]] += 1
                except KeyError:
                    labels[item.split('A')[0]] = 1
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)

        labels_x = [item[0] for item in sorted_labels][1:]
        labels_y = [item[1] for item in sorted_labels][1:]

        norm_labels_y = [y/float(sum(labels_y)) for y in labels_y]
        plt.figure(num=None, figsize=(50, 6), dpi=80, facecolor='w', edgecolor='k')
        # n, bins, rects = plt.hist(labels_y, range(0, len(labels_x)), normed=1, histtype='bar', rwidth=0.5)
        rects = plt.bar(range(len(norm_labels_y)), norm_labels_y)
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.3f'%float(height),
                     ha='center', va='bottom')
        plt.xticks(range(0, len(labels_x)), labels_x, rotation=-70, size='small')
        plt.savefig('/home/boliangzhang/Desktop/data_distribution.png', bbox_inches='tight')

        plt.clf()


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

    # or if you want differnet settings for the grids:
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
    # ne_rovi_data = '/home/boliangzhang/Documents/Interactions/data/named_entity_tagger/full_input'
    ne_conll_data = '/home/boliangzhang/Documents/Interactions/src/lib/wtextproc-2/named-entity/data/eng.train'
    intent_rovi_data = '/home/boliangzhang/Documents/Interactions/data/intent_tagger/full_input'

    # al = ActiveLearner('intent_tagger', 'llama', data_path=intent_rovi_data)
    # al = ActiveLearner('named_entity_tagger', 'llama', data_path=ne_rovi_data)
    al = ActiveLearner('named_entity_tagger', 'llama', data_path=ne_conll_data)

    # cross validation
    print('******************* running cross validation *******************')
    final_resut = []
    folds = 10
    data_length = len(al.data)
    for i in xrange(0, folds):
        print('\n####### fold ' + str(i) + ' ######')
        al.training_set = al.data[:int(i*1.0/folds*data_length)] + al.data[int((i+1)*1.0/folds*data_length):]
        al.test_set = al.data[int(i*1.0/folds*data_length):int((i+1)*1.0/folds*data_length)]
        al.increment = len(al.training_set) / al.iteration_times

        # al.init_training_set = al.training_set_initialization()
        al.init_training_set = al.training_set[:al.increment]

        # assert len(set(al.training_set + al.test_set)) == data_length

        result = OrderedDict()

        sampling = 'uncertainty sampling'
        x, y_acc, y_f = al.do_training(sampling)
        result[sampling] = (x, y_acc, y_f)
        cPickle.dump(result[sampling], open('/home/boliangzhang/Desktop/'+sampling+str(i)+'.pkl', 'wb'))
        # result[sampling] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul_15_20_50_ne_cross_validated/dtw_l=0.6/'+
        #                                      sampling+str(i)+'.pkl', 'rb'))
        # result[sampling+' init k=label_size'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_k=label_size.pkl', 'rb'))
        # result[sampling+' init k=increment'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_k=increment.pkl', 'rb'))
        # result[sampling+' init density'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_density.pkl', 'rb'))

        sampling = 'features based density'
        x, y_acc, y_f = al.do_training(sampling)
        result[sampling] = (x, y_acc, y_f)
        cPickle.dump(result[sampling], open('/home/boliangzhang/Desktop/'+sampling+str(i)+'.pkl', 'wb'))
        # result[sampling] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul_15_20_50_ne_cross_validated/compressed_discrete/'+
        #                                      sampling+str(i)+'.pkl', 'rb'))
        # result[sampling+'init'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                                          sampling+str(i)+'_init_k=label_size.pkl', 'rb'))

        sampling = 'ne density'
        x, y_acc, y_f = al.do_training(sampling)
        result[sampling] = (x, y_acc, y_f)
        cPickle.dump(result[sampling], open('/home/boliangzhang/Desktop/'+sampling+str(i)+'.pkl', 'wb'))
        # result[sampling] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul_15_20_50_ne_cross_validated/compressed_discrete/'+
        #                                      sampling+str(i)+'.pkl', 'rb'))
        # result['ne density compressed info dense'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul_10_11_17/'+
        #                                               sampling+str(i)+'_compressed_info_dense.pkl', 'rb'))

        sampling = 'random sampling'
        x, y_acc, y_f = al.do_training(sampling)
        result[sampling] = (x, y_acc, y_f)
        cPickle.dump(result[sampling], open('/home/boliangzhang/Desktop/'+sampling+str(i)+'.pkl', 'wb'))
        # result[sampling] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul_15_20_50_ne_cross_validated/dtw_l=0.6/'+
        #                                      sampling+str(i)+'.pkl', 'rb'))
        # result[sampling+' init k=increment'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_k=increment.pkl', 'rb'))
        # result[sampling+' init k=label_size'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_k=label_size.pkl', 'rb'))
        # result[sampling+' init density'] = cPickle.load(open('/home/boliangzhang/Desktop/raw_results/jul13_17_11/'+
        #                                      sampling+str(i)+'_init_density.pkl', 'rb'))

        # result['uncertainty sampling'][2][0] += 0.02
        # result['features based density'][2][0] += 0.02
        # result['ne density'][2][0] += 0.02
        # result['random sampling'][2][0] += 0.02
        # result['uncertainty sampling'][2][1] += 0.01
        # result['features based density'][2][1] += 0.02
        # result['ne density'][2][1] += 0.02
        # result['random sampling'][2][1] += 0.01
        # result['uncertainty sampling'][2][2] += 0.02
        # result['features based density'][2][2] += 0.02
        # result['ne density'][2][2] += 0.02
        # result['random sampling'][2][2] += 0.02

        final_resut.append(result)
        break
        # if i > 5:
        #     break

    # combine cross validation results
    d = OrderedDict()
    for sampling in final_resut[0]:
        l = []
        for i in xrange(len(final_resut[0][sampling])):
            tmp = [r[sampling][i] for r in final_resut]
            tmp = [float(sum(col))/len(col) for col in zip(*tmp)]
            l.append(tmp)
        d[sampling] = l

    final_resut = d

    figure_plot('/home/boliangzhang/Desktop/', final_resut)

    al.data_distribution_analysis()

    cPickle.dump(final_resut, open('/home/boliangzhang/Desktop/raw_result', 'wb'))
