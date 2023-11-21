import sqlite3
import streamlit as st
import pandas as pd
import hashlib

conn = sqlite3.connect('GUI\streamlit\Database\Login App\data.db')
c = conn.cursor()


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    else:
        return False


def create_user_table():
    c.execute('CREATE TABLE IF NOT EXISTS userstable (username TEXT, password TEXT)')


def add_user_data(username, password):
    if c.execute('SELECT * FROM userstable WHERE username = (?)', (username, )).fetchone() is not None:
        return -1
    else:
        c.execute('INSERT INTO userstable (username, password) VALUES (?, ?)',
                  (username, password))
        conn.commit()
        return 1


def delete_user_data(username, password):
    if c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?',
                 (username, password)).fetchone() is not None:
        c.execute(
            'DELETE FROM userstable WHERE username = ? AND password = ?', (username, password))
        conn.commit()
        return 1
    else:
        return -1


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def view_all_user():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def main():
    st.title('Simple Login App')

    menu = ['Home', 'Login', 'SignUp', 'Delete']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
    elif choice == 'Login':
        st.subheader('Login Section')
        username = st.sidebar.text_input('User name')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.checkbox('Login'):
            create_user_table()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.success('Welcome Dear {} !!!'.format(username))
                task = st.selectbox(
                    'Task', ['Add Post', 'Analytics', 'Profiles'])
                if task == 'Add Post':
                    st.subheader('Add your post')
                elif task == 'Analytics':
                    st.subheader('Analytics')
                elif task == 'Profiles':
                    st.subheader('User Profiles')
                    user_result = view_all_user()
                    clean_db = pd.DataFrame(user_result, columns=[
                                            'Username', 'Password'])
                    st.dataframe(clean_db)
            else:
                st.warning('Incorrect Username / Password')

    elif choice == 'SignUp':
        st.subheader('Create New Account')
        new_user = st.text_input('Username')
        new_password = st.text_input('Password', type='password')

        if st.button('SignUp'):
            create_user_table()
            res = add_user_data(new_user, make_hashes(new_password))
            if res != -1:
                st.success('You have Successfully created a Valid account')
                st.info('Go to Login Menu to login')
            elif res == -1:
                st.warning(
                    'The Username has Already existed, Please Try another one !!!')
    elif choice == 'Delete':
        st.subheader('Delete an Account')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if st.button('Delete'):
            create_user_table()
            res = delete_user_data(username, make_hashes(password))
            if res == 1:
                st.success('You have Successfully deleted a Valid account')
                st.info('Go to View the new Database')
            elif res == -1:
                st.warning('The account you try to Delete is not existed !!')


if __name__ == '__main__':
    main()
