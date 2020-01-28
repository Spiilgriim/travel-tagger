from sys import argv, exit
from getopt import getopt, GetoptError
from os import listdir
from os.path import isfile, join
import auxFuncs as aux
import config


def main(argv):
    help_content = "\n main.py [--fetch_website][--location_tag][--clustering][--topic_modeling][--label_data][--naive_bayes][--svm_classifier][-random_forest][-h]\n\n --fetch_website url: fetch the content of a website by web crawling, will store it in the articles folder with automatically generated name\n --location_tag: find location tags of articles in the folder article_path will by default use the whole articles folder as path\n --clustering n_clusters: cluster articles with articles TD-IDF as feature will by default use the whole articles folder as path (modify in the config file)\n --topic_modeling model_name: perform topic modeling with articles TD-IDF as feature will by default use the whole articles folder as path (modify in the config file)\n --label_data data_number: Label data_number article selected randomly from the folder article_path (modify in the config file) write it at labeled_data.csv (modify target_ path in the config file)\n --naive_bayes labeled_csv word_embedding: perform naive bayes classification on data expect csv path to labeled data. Word embedding can not take value word2vec\n --svm_classifier labeled_csv word_embedding: perform svm classification on data expect csv path to labeled data. Word embedding can take either tdIdf or word2vec value (change in config file)\n --random_forest labeled_csv word_embedding: perform random forest classification on data expect csv path to labeled data. Word embedding can take either tdIdf or word2vec value (change in config file)\n -h, --help: displays this help\n"
    execute = []
    try:
        opts, args = getopt(argv, "h", [
                            "help", "fetch_website=", "location_tag", "clustering=", "topic_modeling=", "label_data=", "naive_bayes=", "svm_classifier=", "random_forest="])
    except GetoptError:
        print('Invalid argument')
        exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help_content)
            exit()
        else:
            execute.append((opt, arg))

    for program in execute:
        if program[0] == '--fetch_website':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            aux.explore_website(program[1], config.banlist)
        if program[0] == '--clustering':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            blogpaths = config.folder_path
            only_files = [join(blogpaths, f) for f in listdir(
                blogpaths) if isfile(join(blogpaths, f))]
            if config.preprocess_version == 2:
                article_content, article_list = aux.serialize_pre_process2(
                    only_files, clustering=True, location=True)
            else:
                article_content, article_list = aux.serialize_pre_process(
                    only_files, clustering=True, location=True)
            aux.clustering(article_content, article_list, int(program[1]))
        if program[0] == '--location_tag':
            if program[1] != '':
                print('Invalid argument')
                exit(2)
            blogpaths = config.folder_path
            only_files = [join(blogpaths, f) for f in listdir(
                blogpaths) if isfile(join(blogpaths, f))]
            if config.preprocess_version == 2:
                article_content, article_list = aux.serialize_pre_process2(
                    only_files, location=True)
            else:
                article_content, article_list = aux.serialize_pre_process(
                    only_files, location=True)
            aux.location_tag(article_content, article_list)
        if program[0] == '--topic_modeling':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            blogpaths = config.folder_path
            only_files = [join(blogpaths, f) for f in listdir(
                blogpaths) if isfile(join(blogpaths, f))]
            if config.preprocess_version == 2:
                article_content = aux.serialize_pre_process2(
                    only_files)
            else:
                article_content = aux.serialize_pre_process(
                    only_files)
            aux.topic_modeling(article_content, program[1])
        if program[0] == '--label_data':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            blogpaths = config.folder_path
            only_files = [join(blogpaths, f) for f in listdir(
                blogpaths) if isfile(join(blogpaths, f))]
            aux.label_data(only_files, int(program[1]), config.target_path)
        if program[0] == '--naive_bayes':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            aux.naive_bayes(program[1], config.word_embedding)
        if program[0] == '--svm_classifier':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            aux.SVM(program[1], config.word_embedding)
        if program[0] == '--random_forest':
            if program[1] == '':
                print('Invalid argument')
                exit(2)
            aux.random_forest(program[1], config.word_embedding)


if __name__ == "__main__":
    main(argv[1:])
