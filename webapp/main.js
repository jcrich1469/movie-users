// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries
import os 

FIREBASE_API_KEYI = os.getenv('FIREBASE_API_KEY') 
if firebase_api_key is not None:
    print("Yarr, we've got the key to the treasure.")
else:
    print("Blimey! The key's not where it should be. Set it as an environment variable, ye landlubber.")

// Your web app's Firebase configuration
const firebaseConfig = {
	apiKey: FIREB_API_KEY, authDomain: "holocenedb-89d71.firebaseapp.com",
	projectId: "holocenedb-89d71", 
	storageBucket: "holocenedb-89d71.appspot.com", 
	messagingSenderId: "1021615895865", 
	appId: "1:1021615895865:web:97951fd0001696c102e4e8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
