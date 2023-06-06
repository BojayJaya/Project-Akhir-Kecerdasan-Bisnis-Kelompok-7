import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Klasifikasi Penerima PIP dan KIP",
    page_icon='blood.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.write("""<h1>Aplikasi Klasifikasi Penerima PIP dan KIP</h1>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = st.selectbox(
            'Menu',
            ['Preprocessing', 'Modeling', 'Implementation'],
            index=0
        )

    if selected == 'Preprocessing':
        st.subheader('Normalisasi Data')
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')
        
        st.subheader('Data Asli')
        st.dataframe(df, width=600)

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        st.subheader('Data Setelah Normalisasi')
        st.dataframe(scaled_df, width=600)

    elif selected == 'Modeling':
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        y_pred = gaussian.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader('Hasil Modeling dengan Naive Bayes')
        st.write('Akurasi: {:.2f}%'.format(accuracy * 100))

    elif selected == 'Implementation':
        st.subheader('Implementasi Prediksi Penerima PIP dan KIP')
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        X = df.drop(columns=['Status'])
        y = df['Status'].values
        # Encoding label teks menjadi numerik
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)

        st.subheader('Implementasi Prediksi Penerima PIP dan KIP')

        nama = st.number_input('Masukkan nama:')
        jenis_tinggal = st.number_input('Masukkan jenis tinggal:')
        jenjang_pendidikan_ortu_wali = st.number_input('Masukkan jenjang pendidikan ortu atau wali:')
        pekerjaan_ortu_wali = st.number_input('Masukkan pekerjaan ortu atau wali:')
        penghasilan_ortu_wali = st.number_input('Masukkan penghasilan ortu atau wali:')

        input_data = np.array([
            nama,
            jenis_tinggal,
            jenjang_pendidikan_ortu_wali,
            pekerjaan_ortu_wali,
            penghasilan_ortu_wali
        ]).reshape(1, -1)

        input_data_scaled = scaler.transform(input_data)

        prediction = gaussian.predict(input_data_scaled)

        # Menerjemahkan kembali label numerik menjadi label teks
        predicted_label = label_encoder.inverse_transform(prediction)

        st.subheader('Hasil Prediksi')
        st.write('Klasifikasi:', predicted_label[0])
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Anemia Classification",
    page_icon='blood.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Aplikasi Klasifikasi Penderita Anemia</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://lh3.googleusercontent.com/a/ALm5wu2PukBXPMX88VuehLVmYvtTCLj1-XFDgkoky1-JBg=s192-c-rg-br100" width="90" height="90"><br> MUHAMMAD HANIF SANTOSO <p>200411100078</p></h3>""",unsafe_allow_html=True), 
        ["Home", "Description", "Dataset", "Prepocessing", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#FF4B4B"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#FF4B4B"}
            }
        )
        st.write("""
        <div style = "position: fixed; left:40px; bottom: 10px;">
            <center><a href="https://github.com/HanifSantoso05/Aplikasi-Web-Klasifikasi-Penyakit-Anemia"><span><img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" width="40px" height="40px"></span></a><a style = "margin-left: 20px;" href="http://hanifsantoso05.github.io/datamining/intro.html"><span><img src="https://friconix.com/png/fi-stluxx-jupyter-notebook.png" width="40px" height="40px"></span></a> <a style = "margin-left: 20px;" href="mailto: hanifsans05@gmail.com"><span><img src="https://cdn-icons-png.flaticon.com/512/60/60543.png" width="40px" height="40px"></span></a></center>
        </div> 
        """,unsafe_allow_html=True)

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://cdn-2.tstatic.net/jatim/foto/bank/images/anemia.jpg" width="500" height="300">
        </h3>""",unsafe_allow_html=True)
        st.write("""
        Anemia adalah suatu kondisi di mana Anda kekurangan sel darah merah yang sehat untuk membawa oksigen yang cukup ke jaringan tubuh Anda. Penderita anemia, juga disebut hemoglobin rendah, bisa membuat Anda merasa lelah dan lemah.
        """)

    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""
        Dataset ini merupakan data gejala-gejala penderita anemia yang terdapat di website kaggle.com, data ini nantinya di gunakan untuk melakukan prediksi penyakit anemia. Dataset ini sendiri terdiri dari 6 atribut yaitu Gender, Hemoglobin, MCHC, MCV, MCH dan Hasil.
        """)

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Dataset ini digunakan untuk melakukan klasifikasi penderita penyakit anemia. Setelah dilakukan klasifikasi selanjutnya dilakukan implementasi dengan memprediksi pasien apakah mengidap anemia atau tidak.
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            - Jenis kelamin: 
                - 0 : laki-laki
                - 1 : perempuan.
            - Hemoglobin: Hemoglobin adalah protein dalam sel darah merah Anda yang membawa oksigen ke organ dan jaringan tubuh Anda dan mengangkut karbon dioksida dari organ dan jaringan Anda kembali ke paru-paru.
                - Normal Range
                    - Wanita Dewasa : 12.0 – 15.5 g/dL.
                    - Pria Dewasa : 13.5 – 17.5 g/dL.
                    - Anak - anak : 11.0 - 13,5 g/dL.
                    - Bayi (3 Bulan) : 9.5 - 12.5 g/dL.
                    - Bayi Baru Lahir : 15.0 - 21.0 g/dL.
            - MCH: MCH atau mean corpuscular hemoglobin adalah pengukuran yang menjelaskan jumlah rata-rata hemoglobin dalam satu sel darah merah (eritrosit).
                - Normal Range : 26 – 33 pg.
            - MCHC: MCHC adalah singkatan dari rata-rata konsentrasi hemoglobin corpuscular. Ini adalah ukuran konsentrasi rata-rata hemoglobin di dalam satu sel darah merah.
                - Normal Range untuk orang Dewasa : 32-36%.
                - Normal Range untuk Bayi Baru Lahir : 31-35%.
            - MCV: MCV adalah singkatan dari mean corpuscular volume. Tes darah MCV mengukur ukuran rata-rata sel darah merah Anda.
                - Normal Range
                    - Dewasa : 80 – 100 fL.
                    - Bayi baru lahir : 98 – 122 fL.
                    - Anak usia 1-3 tahun : 73 – 101 fL.
                    - Anak usia 4-5 tahun : 72 – 88 fL.
                    - Anak usia 6-10 tahun : 69 – 93 fL.
            - Hasil: 
                - 0 : Negative Anemia
                - 1 : Positive Anemia.
            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset">Klik disini</a>""", unsafe_allow_html=True)
        
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset anemia ini adalah NUMERICAL.
        """)

    elif selected == "Dataset":
        st.subheader("""Dataset Anemia""")
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')
        st.dataframe(df, width=600)

    elif selected == "Prepocessing":
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Status'])
        y = df['Status'].values
        df_min = X.min()
        df_max = X.max()
        
        #NORMALISASI NILAI X
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        st.subheader('Hasil Normalisasi Data')
        st.dataframe(scaled_features, width=600)

        st.subheader('Target Label')
        dumies = pd.get_dummies(df.Status).columns.values.tolist()
        dumies = np.array(dumies)

        labels = pd.DataFrame({
            'Positive' : [dumies[1]],
            'Negative' : [dumies[0]]
        })

        st.write(labels)

    elif selected == "Modeling":
        #Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        #Preprocessing data
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Status'])
        y = df['Status'].values
        
        #NORMALISASI NILAI X
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        #Split Data 
        training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            naive = st.checkbox('Gaussian Naive Bayes')
            submitted = st.form_submit_button("Submit")

            #Gaussian Naive Bayes
            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)
            # prediction
            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()
            #Accuracy
            gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

            if submitted :
                if naive :
                    st.write('Model Naive Bayes accuracy score: {0:0.0f}'. format(gaussian_akurasi),'%')

            grafik = st.form_submit_button("Grafik akurasi semua model")
            if grafik:
                data = pd.DataFrame({
                    'Akurasi' : [gaussian_akurasi],
                    'Model' : ['Gaussian Naive Bayes'],
                })

                bar_chart = px.bar(data, 
                    x='Model', 
                    y='Akurasi',
                    text='Akurasi',
                    color_discrete_sequence =['#FF4B4B']*len(data),
                    width=680)
                bar_chart

    elif selected == "Implementation":
        with st.form("Implementation"):
            #Read Dataset
            df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

            #Preprocessing data
            #Mendefinisikan Varible X dan Y
            X = df.drop(columns=['Status'])
            y = df['Status'].values
            
            #NORMALISASI NILAI X
            scaler = MinMaxScaler()
            #scaler.transform(features)
            scaled = scaler.fit_transform(X)
            features_names = X.columns.copy()
            #features_names.remove('label')
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            #Split Data 
            training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

            #Gaussian Naive Bayes
            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)
            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()

            st.subheader("Implementasi Prediksi Penyakit Diabetes")
            nama = st.number_input('Masukkan nama:')
            jenist_inggal = st.number_input('Masukkan jenis jinggal:')
            jenjang_pendidikan_ortu_wali = st.number_input('Masukkan jenis pendidikan ortu atau wali:')
            pekerjaan_ortu_wali = st.number_input('Masukkan pekerjaan ortu atau wali:')
            penghasilan_ortu_wali = st.number_input('Masukkan penghasilan ortu atau wali:')
            model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                    ('Gaussian Naive Bayes'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    nama,
                    jenist_inggal,
                    jenjang_pendidikan_ortu_wali,
                    pekerjaan_ortu_wali,
                    penghasilan_ortu_wali
                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                    akurasi = round(100 * accuracy_score(test_label,probas))

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :',model)
                st.write('Akurasi: {0:0.0f}'. format(akurasi),'%')
                

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')
