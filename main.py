# PACKAGES
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
import graphviz
import pandas as pd
import requests
from sklearn.tree import export_graphviz
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None


def main():
    # DATA CLEANING
    s1 = pd.read_csv('student-mat.csv')
    s2 = pd.read_csv('student-por.csv')
    all_students = pd.concat([s1, s2], axis=0)
    all_students.dropna()
    all_students.dropna()
    all_students['Harmful_Consumption'] = 0

    all_students.loc[((all_students['Dalc'] + all_students['Walc'])
                      >= 6), 'Harmful_Consumption'] = 1
    all_students.to_csv('Data.csv')

# QUESTION 1
    #rt = ["famsize", "Pstatus", "studytime","famsup", "Medu", "famrel","Fedu", "Mjob","Fjob", "nursery","studytime", "failures", "goout", "absences", "G1", "G2", "G3" ]
    ret = ["famsize", "studytime", "famsup", "Medu", "famrel", "Fedu", "Dalc",
           "freetime", "goout", "Walc", "absences", "G1", "G2", "G3", "nursery", "higher"]
    my_dat = all_students.loc[:, ret]

    # "nursery","higher"
    dff = my_dat[my_dat["famsize"] == "LE3"]
    ne_dt = dff[dff["nursery"] == "no"]
    print(ne_dt.shape[0])
    dff = my_dat[my_dat["famsize"] == "LE3"]
    ne_dt1 = dff[dff["nursery"] == "yes"]
    ne_dt1.shape[0]

    def famsup_mean(dt):
        """
        This function takes the data and returns the mean of students study time
        """
        my_dat_1 = dt[dt["famsup"] == "no"]
        ft1 = my_dat_1["studytime"].mean()
        print("The mean study time for student without family support {}".format(ft1))
        my_dat_2 = dt[dt["famsup"] == "yes"]
        ft2 = my_dat_2["studytime"].mean()
        print("The mean study time for student with family support {}".format(ft2))

    df_b = my_dat.iloc[1:6, 1:4]
    dfd = df_b[df_b["famsup"] == "no"]
    print("The actual mean {}".format(dfd["studytime"].mean()))
    print("The mean from famsup_mean function:")
    famsup_mean(dfd)

    famsup_mean(my_dat)

    def famsz_study_time(dt):
        """
        This function takes the data, manipulate it and returns
        the mean of study time for students from different family background
        """
        my_dat_fs1 = dt[dt["famsize"] == "LE3"]
        print("The mean study time for student with less family size {}".format(
            my_dat_fs1["studytime"].mean()))
        my_dat_fs2 = dt[dt["famsize"] == "GT3"]
        print("The mean study time for student with large family size {}".format(
            my_dat_fs2["studytime"].mean()))

    dff = my_dat.iloc[:10, :5]
    dff1 = dff[dff["famsize"] == "LE3"]
    dff2 = dff[dff["famsize"] == "GT3"]
    print("Testing one data actual mean is {}".format(
        dff1["studytime"].mean()))
    print("Testing two data actual mean is {}".format(
        dff2["studytime"].mean()))
    print("The both testing data mean from famsz_study_time function repectively:")
    famsz_study_time(dff)

    famsz_study_time(my_dat)

    my_dat_f1 = my_dat[(my_dat["famsize"] == "LE3")
                       & (my_dat["famsup"] == "no")]
    my_dat_f2 = my_dat[(my_dat["famsize"] == "LE3") &
                       (my_dat["famsup"] == "yes")]
    my_dat_f3 = my_dat[(my_dat["famsize"] == "GT3") &
                       (my_dat["famsup"] == "yes")]
    my_dat_f4 = my_dat[(my_dat["famsize"] == "GT3")
                       & (my_dat["famsup"] == "no")]
    st_per = my_dat_f2.shape[0]/(my_dat_f1.shape[0] + my_dat_f2.shape[0])*100
    print("Percentage of students with small family size that get family support, {}".format(st_per))
    st_per_rt = my_dat_f3.shape[0] / \
        (my_dat_f3.shape[0] + my_dat_f4.shape[0])*100
    print("Percentage of students with large family size that get family support, {}".format(st_per_rt))
    print("""The above percntage shows that family size has some impact on the student
          sucess, especially if the students have sibling, they will more likely to study more
          This could be there will be enough family member to help the student""")

    rt = ["famsize", "studytime", "famsup", "Medu", "famrel",
          "Fedu", "Dalc", "Walc", "absences", "G1", "G2", "G3"]
    my_dat = all_students.loc[:, rt]
    ty = ["G1", "G2", "G3"]
    new_dt = my_dat[ty]
    new_dt.mean(axis=1)
    my_dat["Mean_Grade"] = new_dt.mean(axis=1)

    def hist_func(dt, N_pt, n):
        """
        This function takes data, number of point, and bins and
        returns the histogram plot of mean grade score students from different family size
        """
        N_points = 1000
        n_bins = 20
        x = my_dat[my_dat["famsize"] == "LE3"]
        x = x["Mean_Grade"]
        y = my_dat[my_dat["famsize"] == "GT3"]
        y = y["Mean_Grade"]
        fig, axs = plt.subplots(
            1, 2, sharey=True, figsize=(15, 9), tight_layout=True)
        # We can set the number of bins with the `bins` kwarg
        axs[0].hist(x, bins=n_bins)
        axs[0].set_title("The student from smaller family size Mean_Grade")
        #
        axs[1].hist(y, bins=n_bins)
        axs[1].set_title("The student from larger family size Mean_Grad")
        axs[0].grid()
        axs[1].grid()

    hist_func(my_dat, 1000, 20)

    def scat_plot_func(dt):
        plt.scatter(my_dat["studytime"], my_dat["Mean_Grade"], s=my_dat["G3"])
        plt.xlabel("Study Time")
        plt.ylabel("Mean Grade Score")
        plt.title("The Scatter Plot Study Time vs Mean Grade Score")
        plt.grid()

    scat_plot_func(my_dat)

    def famedu_grad_plt(dt):
        """
        This function takes the data and generate a line
        plot for parents education level and mean grade score
        """
        ne_data = dt.drop(["famsize", "famsup"], axis=1)
        sns.relplot(x="Fedu", y="Mean_Grade", data=ne_data, kind="line")
        plt.title('Father education vs. mean period grade')
        plt.ylabel("mean period grade")
        plt.grid()
        sns.relplot(x="Medu", y="Mean_Grade", data=ne_data, kind="line")
        plt.title('Mother education vs. second period grade')
        plt.grid()

    famedu_grad_plt(my_dat)

    def modified_plot(dt):
        """
        This function takes the data, modify it and generate a line
        plot for parents education level and mean grade score
        """
        ne_data = dt.drop(["famsize", "famsup"], axis=1)
        new_data = ne_data[(ne_data["Fedu"] != 0) & (ne_data["Medu"] != 0)]
        sns.relplot(x="Fedu", y="Mean_Grade", data=new_data, kind="line")
        plt.title('Father education vs. Mean grade score')
        plt.ylabel("mean period grade")
        plt.grid()
        sns.relplot(x="Medu", y="Mean_Grade", data=new_data, kind="line")
        plt.title('Mother education vs. Mean grade score')
        plt.grid()

    modified_plot(my_dat)

    def lin_plt_famedu_alc(dt):
            # Select the target data
        ne_data = my_dat.drop(["famsize", "famsup"], axis=1)
        # Plotting the sns for the given data
        sns.relplot(x="Fedu", y="Dalc", data=ne_data, kind="line")
        plt.title("Father education vs Weekend alcohol consumption of students")
        plt.grid()  # add the gride
        sns.relplot(x="Fedu", y="Walc", data=ne_data, kind="line")
        plt.title("Father education vs Weekday alcohol consumption of students")
        plt.grid()
        sns.relplot(x="Medu", y="Dalc", data=ne_data, kind="line")
        plt.title("Mother education vs Weekend alcohol consumption of students")
        plt.grid()
        sns.relplot(x="Medu", y="Walc", data=ne_data, kind="line")
        plt.title("Mother education vs Weekday alcohol consumption of students")
        plt.grid()

    lin_plt_famedu_alc(my_dat)

    def alc_absc_pl(dt):
            # select the target data
        ne_data = dt.drop(["famsize", "famsup"], axis=1)
        # plotting command
        sns.relplot(x="Dalc", y="absences", data=ne_data, kind="line")
        # Title the plot
        plt.title("Weekend Alcohol consumption vs Absence of the students")
        plt.grid()  # add the grid to the plot

        sns.relplot(x="Walc", y="absences", data=ne_data, kind="line")
        plt.title("Weekday Alcohol consumption vs Absence of the student")
        plt.grid()

    alc_absc_pl(my_dat)

    ml_data = my_dat.drop(["G1", "G2", "G3"], axis=1)
    X = ml_data.loc[:, ml_data.columns != 'Mean_Grade']
    X = pd.get_dummies(X)
    y = np.round(ml_data['Mean_Grade'])

    # splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)
    # create linear regression object
    reg = linear_model.LinearRegression()

    # train the model using the training sets
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # regression coefficients
    print('Coefficients: \n', reg.coef_)

    # variance score
    print('Testing data Variance score: {}'.format(reg.score(X_test, y_test)))
    print('Training data Variance score : {}'.format(reg.score(X_train, y_train)))
    y_train_pred = reg.predict(X_train)  # generate the predict y values
    y_test_pred = reg.predict(X_test)  # generate the test y values
    # Calculate and printing the MSE and R^2

    print("Traing data Mean squared error {}".format(
        mean_squared_error(y_train, y_train_pred)))
    print("Testing data Mean squared error {}".format(
        mean_squared_error(y_test, y_pred)))
    print("r_2 statistic:{}".format(r2_score(y_test, y_test_pred)))

# QUESTION 2
    all_students  = pd.read_csv('data.csv')
    harmed = all_students[all_students['Harmful_Consumption'] == 1]
    healthy = all_students[all_students['Harmful_Consumption'] == 0]

    average_student_study = all_students['studytime'].mean()
    harmful_consumer_study = harmed['studytime'].mean()
    names = ['Harmful consumption', 'Average']
    values = [harmful_consumer_study, average_student_study]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(1, 4)
    plt.ylabel("scale 1-4")
    plt.suptitle('Average amount of study time of both groups. Scale 1-4  ')
    plt.show()

    father = all_students[all_students['guardian'] == 'father']
    mother = all_students[all_students['guardian'] == 'mother']
    other  = all_students[all_students['guardian'] == 'other']
    fnum = len(father[father['Harmful_Consumption'] == 1]) / len(father)
    mnum = len(mother[mother['Harmful_Consumption'] == 1]) / len(mother)
    onum = len(other[other['Harmful_Consumption'] == 1]) / len(other)
    names = ['Father', 'Mother', 'other']
    values = [fnum * 100, mnum * 100, onum * 100]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(0, 50)
    plt.ylabel("Percentage")
    plt.suptitle('Percentage of those who consume harmful amounts of alcohol by who their guardian is')
    plt.show()

    parent_edu = all_students['Medu'].mean() + all_students['Fedu'].mean() -1
    parent_edu_harmed = harmed['Medu'].mean() + harmed['Fedu'].mean() -1
    names = ['Harmful consumption', 'Average']
    values = [parent_edu_harmed , parent_edu]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(1, 7)
    plt.ylabel("scale 1-7")
    plt.suptitle('Average parental education of both groups. Scale 1-7  ')
    plt.show()

    urban_ratio_all = len(all_students[all_students['address'] == 'R']) / len(all_students)
    urban_ratio_harmed = len(harmed[harmed['address'] == 'R']) / len(harmed)
    names = ['Harmful consumption', 'All']
    values = [round(urban_ratio_harmed * 100) , round(urban_ratio_all * 100)]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(0, 50)
    plt.ylabel("Percentage")
    plt.xlabel("Those with harmuful alcohol consumption vs average")
    plt.suptitle('Pertentage of Students in each group who live in rural areas')
    plt.show()

    gender_rat = len(all_students[all_students['sex'] == 'M']) / len(all_students)
    gender_rat_harmed = len(harmed[harmed['sex'] == 'M']) / len(harmed)
    names = ['Harmful consumption', 'All']
    values = [round(gender_rat_harmed * 100), round(gender_rat * 100)]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.suptitle('Pertentage of Students in each group who are Male')
    plt.show()

    healthy_fails = healthy['failures'].mean()
    harm_fail = harmed['failures'].mean()
    names = ['Harmful consumption', 'Those who don\'t consume harmful amoutns of alcohol']
    values = [harm_fail,healthy_fails]
    plt.figure(figsize=(15, 7))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("average amount of classes faild")
    plt.suptitle('Average amount of classes failed by group')
    plt.show()


# QUESTION 3
    def plot_tree(model, X, y):
        """
        This function was taken from Lecture #9, it graphs the decision classifier in the form of a tree.
        """
        dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, class_names=y.unique(
        ), filled=True, rounded=True, special_characters=True)
        return graphviz.Source(dot_data)

    dfn = pd.read_csv('Data.csv')
    del dfn['Unnamed: 0']
    del dfn['school']
    del dfn['Walc']
    del dfn['Dalc']
    dfn['Harmful_Consumption'] = dfn['Harmful_Consumption'].map(
        {1: 'yes', 0: 'no'})
    family_data = dfn[['famsize', 'Pstatus', 'Medu', 'Fedu',
                       'Mjob', 'Fjob', 'famsup', 'famrel', 'Harmful_Consumption']]
    school_data = dfn[['traveltime', 'studytime', 'failures', 'absences',
                       'schoolsup', 'freetime', 'G1', 'G2', 'G3', 'Harmful_Consumption']]

    # model based on family data includes constants such as age, sex & health
    fam_x = family_data.loc[:, family_data.columns != 'Harmful_Consumption']
    fam_x = pd.get_dummies(fam_x)
    fam_y = dfn['Harmful_Consumption']
    family_model = DecisionTreeClassifier()
    family_model.fit(fam_x, fam_y)
    fam_tree = plot_tree(family_model, fam_x, fam_y)
    fam_tree.render('fam_tree.png', view=True)

    famX_train, famX_test, famy_train, famy_test = train_test_split(
        fam_x, fam_y, test_size=0.2)
    family_model.fit(famX_train, famy_train)
    # accuracy at predicting alcoholic student based off of family characteristics
    fam_acc_score = accuracy_score(famy_test, family_model.predict(famX_test))
    print('There is', round(fam_acc_score * 100),
          "percent accuracy when predicting if a student will harmfully consume alcohol based off family related features.")

    # model based on school data includes constants such as age, sex & health
    school_x = school_data.loc[:, school_data.columns != 'Harmful_Consumption']
    school_x = pd.get_dummies(school_x)
    school_y = dfn['Harmful_Consumption']
    school_model = DecisionTreeClassifier()
    school_model.fit(school_x, school_y)
    skool_tree = plot_tree(school_model, school_x, school_y)
    skool_tree.render('school_tree', view=True)

    schoolX_train, schoolX_test, schooly_train, schoooly_test = train_test_split(
        school_x, school_y, test_size=0.2)
    school_model.fit(schoolX_train, schooly_train)

    # accuracy at predicting alcoholic student based off of family characteristics
    school_acc_score = accuracy_score(
        schoooly_test, school_model.predict(schoolX_test))
    print('There is', round(school_acc_score * 100),
          "percent accuracy when predicting if a student will harmfully consume alcohol based off school related features.")

    knn_data = dfn[['studytime', 'paid',
                   'activities', 'romantic', 'famrel', 'goout']]

    for col_name in knn_data.columns:
        if(knn_data[col_name].dtype == 'object'):
            knn_data[col_name] = knn_data[col_name].astype('category')
            knn_data[col_name] = knn_data[col_name].cat.codes

    def knn_and_plot(feature_1, feature_2):
        """
        This function runs the data through a KNN classifier as well as graphs the results.
        """
        y = 1*(dfn.Harmful_Consumption == "no") + \
            2*(dfn.Harmful_Consumption == "yes")
        x = knn_data[[feature_1, feature_2]].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        m = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
        m.fit(x_train, y_train)

        y_pred = m.predict(x_test)
        accuracy = round(accuracy_score(y_test, y_pred), 3) * 100
        accuracy_statement = "There is {} percent accuracy when using a K-NN classifier with features ({}, {}) and a K value of 4.".format(
            accuracy, feature_1, feature_2)

        range1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
        range2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
        (xx1, xx2) = np.meshgrid(range1, range2)
        Xgrid = np.column_stack((xx1.ravel(), xx2.ravel()))
        yhat = m.predict(Xgrid)
        plt.figure(figsize=(10, 10))

        plt.imshow(yhat.reshape((100, 100)),
                   alpha=0.3, extent=[xx1.min(), xx1.max(), xx2.min(), xx2.max()],
                   origin='lower', aspect='auto')

        plot = plt.scatter(x[:, 0], x[:, 1], c=y, s=100,
                           alpha=0.5, edgecolor="k")
        plt.title(accuracy_statement)

        return plot

    features = list(combinations(list(knn_data.columns), 2))
    for combination in features:
        knn_and_plot(combination[0], combination[1])


if __name__ == "__main__":
    main()
