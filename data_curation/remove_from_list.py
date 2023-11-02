import pandas as pd


if __name__ == "__main__":
    """
    Remove cases using removal list
    """
    all_cases_list = '/home/bella/Phd/docs/clinical/abnormal_cases/all_cases_including_abnormal.csv'
    normal_iugr_list = '/home/bella/Phd/docs/clinical/normal_and_IUGR_cases.csv'
    abnormal_cases_path = '/home/bella/Phd/docs/clinical/abnormal_cases/abnormal_cases.csv'
    all_cases_pd = pd.read_csv(all_cases_list)
    all_subjects_lst = all_cases_pd['Subject'].tolist()
    normal_iugr_set = set(pd.read_csv(normal_iugr_list)['Subject'].tolist())

    num_removed = 0
    for subject in all_subjects_lst:
        if subject in normal_iugr_set:
            print('removing case ' + subject)
            num_removed += 1
            all_cases_pd = all_cases_pd.drop(all_cases_pd[all_cases_pd['Subject']==subject].index)

    print('number of removed cases: ' + str(num_removed))
    all_cases_pd.to_csv(abnormal_cases_path)