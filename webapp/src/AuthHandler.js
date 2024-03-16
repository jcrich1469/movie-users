import { initializeApp } from "firebase/app";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, onAuthStateChanged, signOut } from "firebase/auth";


const firebaseConfig = {
	apiKey: "AIzaSyA1HthQRuZoZ13VeaPXu0p8tUP44Yy7zMc", 
	authDomain: "holocenedb-89d71.firebaseapp.com",
	projectId: "holocenedb-89d71", 
	storageBucket: "holocenedb-89d71.appspot.com", 
	messagingSenderId: "1021615895865", 
	appId: "1:1021615895865:web:97951fd0001696c102e4e8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
const auth = getAuth(app);

class AuthHandler {
    constructor() {
        this.auth = auth; // Already initialized above
    }

    // Sign up new users
    async signUp(email, password) {
        try {
            const userCredential = await createUserWithEmailAndPassword(this.auth, email, password);
            console.log("User created successfully:", userCredential.user);
            return userCredential.user;
        } catch (error) {
            console.error("Error signing up:", error.code, error.message);
            throw error;
        }
    }

    // Sign in existing users
    async signIn(email, password) {
        try {
            const userCredential = await signInWithEmailAndPassword(this.auth, email, password);
            console.log("User signed in successfully:", userCredential.user);
            return userCredential.user;
        } catch (error) {
            console.error("Error signing in:", error.code, error.message);
            throw error;
        }
    }

    // Sign out the current user
    async signOut() {
        try {
            await signOut(this.auth);
            console.log("User signed out successfully");
        } catch (error) {
            console.error("Error signing out:", error.code, error.message);
            throw error;
        }
    }

    // Monitor authentication state
    monitorAuthState(callback) {
        onAuthStateChanged(this.auth, user => {
            callback(user);
        });
    }
}

export default AuthHandler;
