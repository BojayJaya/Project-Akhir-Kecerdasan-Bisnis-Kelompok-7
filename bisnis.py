from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Klasifikasi",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h3 style="text-align: justify;">KLASIFIKASI PENERIMA BANTUAN PIP DAN KIP SD NEGERI LOMBANG DAJAH 1 MENGGUNAKAN METODE</h3></center>
<center><h3 style="text-align: justify;">NAIVE BAYES, ANN, SVM, LOGISTIC REGRESSION, DAN KNN</h3></center>
""", unsafe_allow_html=True)
st.write("### Dosen Pengampu : Eka Mala Sari Rochman, S.Kom., M.Kom.", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h3 style="text-align: center;"><img src="" width="120" height="120"></h3>""",unsafe_allow_html=True),
            ["Home", "Implementation"],
            icons=['house', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style="text-align: center;">
        <img src="https://bareng-bjn.desa.id/desa/upload/artikel/sedang_1554884848_e.jpg" width="500" height="300">
        </h3>""", unsafe_allow_html=True)

    elif selected == "Implementation":
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/datasetpipkip.csv')

        X = df[['Jenis_Tinggal', 'Jenis_Pendidikan_Ortu_Wali', 'Pekerjaan_Ortu_Wali', 'Penghasilan_Ortu_Wali']]
        y = df['Status'].values

        # One-hot encoding pada atribut kategorikal
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X.astype(str)).toarray()
        feature_names = encoder.get_feature_names_out(input_features=X.columns)
        scaled_features = pd.DataFrame(X_encoded, columns=feature_names)

        # Label encoding pada target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split Data
        training, test, training_label, test_label = train_test_split(scaled_features, y_encoded, test_size=0.1, random_state=42)

        # Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian.fit(training, training_label)
        probas_gaussian = gaussian.predict_proba(test)
        probas_gaussian = probas_gaussian[:, 1]
        probas_gaussian = probas_gaussian.round().astype(int)

        # Artificial Neural Network
        # ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        # Fungsi aktivasi sigmoid
        # ann = MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', max_iter=2000)
        # Fungsi aktivasi sigmoid
        # ann = MLPClassifier(hidden_layer_sizes=(200, 100), activation='sigmoid', max_iter=2000)
        # Fungsi aktivasi tanh
        # ann = MLPClassifier(hidden_layer_sizes=(200, 100), activation='tanh', max_iter=2000)
        ann = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000)
        ann.fit(training, training_label)
        probas_ann = ann.predict_proba(test)
        probas_ann = probas_ann[:, 1]
        probas_ann = probas_ann.round().astype(int)


        # Support Vector Machine
        # kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        kernel = 'linear'  # Ganti dengan kernel yang diinginkan, misalnya 'poly', 'rbf', atau 'sigmoid'
        svm = SVC(kernel=kernel, probability=True)
        svm.fit(training, training_label)
        probas_svm = svm.predict_proba(test)
        probas_svm = probas_svm[:, 1]
        probas_svm = probas_svm.round().astype(int)

        # Logistic Regression
        # ubah dengan ini 
        # solver='liblinear', solver='newton-cg', solver='lbfgs', solver='sag', solver='saga'
        logistic_regression = LogisticRegression(solver='liblinear')
        logistic_regression.fit(training, training_label)
        probas_logistic_regression = logistic_regression.predict_proba(test)
        probas_logistic_regression = probas_logistic_regression[:, 1]
        probas_logistic_regression = probas_logistic_regression.round().astype(int)

        # K-Nearest Neighbors
        k = 5  # Nilai K default jika ingin merubah tinggal ubah nilai k nya
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training, training_label)
        probas_knn = knn.predict_proba(test)
        probas_knn = probas_knn[:, 1]
        probas_knn = probas_knn.round().astype(int)

        # Decision Tree dengan kriteria entropy
        decision_tree = DecisionTreeClassifier(criterion='gini')
        # decision_tree = DecisionTreeClassifier(criterion='entropy')
        decision_tree.fit(training, training_label)
        probas_decision_tree = decision_tree.predict_proba(test)
        probas_decision_tree = probas_decision_tree[:, 1]
        probas_decision_tree = probas_decision_tree.round().astype(int)

        st.subheader("Implementasi Penerima bantuan PIP dan KIP")
        jenis_tinggal = st.selectbox('Masukkan jenis tinggal:',
                                    ['Bersama orang tua', 'Wali'])
        jenis_pendidikan_ortu_wali = st.selectbox('Masukkan jenis pendidikan ortu atau wali:',
                                                ['Tidak sekolah', 'SD sederajat', 'SMP sederajat', 'SMA sederajat', 'D2', 'S1'])
        pekerjaan_ortu_wali = st.selectbox('Masukkan pekerjaan ortu atau wali:',
                                        ['Sudah Meninggal', 'Petani', 'Pedagang Kecil', 'Karyawan Swasta', 'Wiraswasta'])
        penghasilan_ortu_wali = st.selectbox('Pilih penghasilan ortu atau wali:',
                                            ['Tidak Berpenghasilan', 'Kurang dari 1.000.000', '500,000 - 999,999', '1,000,000 - 1,999,999'])
        if st.button('Submit'):
            inputs = np.array([
                jenis_tinggal,
                jenis_pendidikan_ortu_wali,
                pekerjaan_ortu_wali,
                penghasilan_ortu_wali
            ]).reshape(1, -1)

            # Ubah input menjadi tipe data string
            inputs = inputs.astype(str)

            # Transformasi one-hot encoding pada input data
            inputs_encoded = encoder.transform(inputs).toarray()

            st.subheader('Hasil Prediksi')

            if len(test_label) > 0:
                test_label = test_label.astype(int)

                # Gaussian Naive Bayes
                input_pred_gaussian = gaussian.predict(inputs_encoded)
                probas_gaussian = probas_gaussian.round().astype(int)
                akurasi_gaussian = round(100 * accuracy_score(test_label, probas_gaussian))
                st.write('Gaussian Naive Bayes')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_gaussian), '%')
                if input_pred_gaussian == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

                # Artificial Neural Network
                input_pred_ann = ann.predict(inputs_encoded)
                probas_ann = probas_ann.round().astype(int)
                akurasi_ann = round(100 * accuracy_score(test_label, probas_ann))
                st.write('Artificial Neural Network')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_ann), '%')
                if input_pred_ann == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

                # Support Vector Machine
                input_pred_svm = svm.predict(inputs_encoded)
                probas_svm = probas_svm.round().astype(int)
                akurasi_svm = round(100 * accuracy_score(test_label, probas_svm))
                st.write('Support Vector Machine')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_svm), '%')
                if input_pred_svm == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

                # Logistic Regression
                input_pred_logistic_regression = logistic_regression.predict(inputs_encoded)
                probas_logistic_regression = probas_logistic_regression.round().astype(int)
                akurasi_logistic_regression = round(100 * accuracy_score(test_label, probas_logistic_regression))
                st.write('Logistic Regression')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_logistic_regression), '%')
                if input_pred_logistic_regression == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

                # K-Nearest Neighbors
                input_pred_knn = knn.predict(inputs_encoded)
                probas_knn = probas_knn.round().astype(int)
                akurasi_knn = round(100 * accuracy_score(test_label, probas_knn))
                st.write('K-Nearest Neighbors')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_knn), '%')
                if input_pred_knn == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

                # Decision Tree
                input_pred_decision_tree = decision_tree.predict(inputs_encoded)
                probas_decision_tree = probas_decision_tree.round().astype(int)
                akurasi_decision_tree = round(100 * accuracy_score(test_label, probas_decision_tree))
                st.write('Decision Tree')
                st.write('Akurasi: {0:0.0f}'.format(akurasi_decision_tree), '%')
                if input_pred_decision_tree == 1:
                    st.error('PIP')
                else:
                    st.success('KIP')

            else:
                st.error('Tidak ada data untuk melakukan prediksi.')


            if len(test_label) > 0:
                test_label = test_label.astype(int)

                # Gaussian Naive Bayes
                input_pred_gaussian = gaussian.predict(inputs_encoded)
                probas_gaussian = probas_gaussian.round().astype(int)
                akurasi_gaussian = round(100 * accuracy_score(test_label, probas_gaussian))

                # Artificial Neural Network
                input_pred_ann = ann.predict(inputs_encoded)
                probas_ann = probas_ann.round().astype(int)
                akurasi_ann = round(100 * accuracy_score(test_label, probas_ann))

                # Support Vector Machine
                input_pred_svm = svm.predict(inputs_encoded)
                probas_svm = probas_svm.round().astype(int)
                akurasi_svm = round(100 * accuracy_score(test_label, probas_svm))

                # Logistic Regression
                input_pred_logistic_regression = logistic_regression.predict(inputs_encoded)
                probas_logistic_regression = probas_logistic_regression.round().astype(int)
                akurasi_logistic_regression = round(100 * accuracy_score(test_label, probas_logistic_regression))

                # K-Nearest Neighbors
                input_pred_knn = knn.predict(inputs_encoded)
                probas_knn = probas_knn.round().astype(int)
                akurasi_knn = round(100 * accuracy_score(test_label, probas_knn))

                # Decision Tree
                input_pred_decision_tree = decision_tree.predict(inputs_encoded)
                probas_decision_tree = probas_decision_tree.round().astype(int)
                akurasi_decision_tree = round(100 * accuracy_score(test_label, probas_decision_tree))

                # Create a bar chart to compare accuracies
                models = ['Gaussian Naive Bayes', 'Artificial Neural Network', 'Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree']
                accuracies = [akurasi_gaussian, akurasi_ann, akurasi_svm, akurasi_logistic_regression, akurasi_knn, akurasi_decision_tree]

                fig = go.Figure(data=[go.Bar(x=models, y=accuracies)])
                fig.update_layout(title='Perbandingan Akurasi Model', xaxis_title='Model', yaxis_title='Akurasi (%)')

                st.plotly_chart(fig)
