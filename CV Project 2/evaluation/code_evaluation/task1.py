import os

def checkResults(results_folder='submission/Task1/', gd_folder='train/Task1/ground-truth/'):
    results = []
    gd = []
    filenames = []

    bad_files = []

    for filename in os.listdir(results_folder):
        f = open(results_folder + filename)
        results.append(f.readlines())
        filenames.append(filename.split("_predicted")[0])

    for filename in os.listdir(gd_folder):
        f = open(gd_folder + filename)
        gd.append(f.readlines())


    correct = 0
    for i in range(len(results)):
        if results[i] == gd[i]:
            correct += 1
        else:
            bad_files.append(filenames[i])

    print(len(results))
    print(correct)
    print(f'{len(bad_files)} bad predictions')
    print(f'Accuracy is {correct / len(results) * 100}%')

    return bad_files

print(f'Bad predictions at: {checkResults()}')