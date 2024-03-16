// Import the necessary functions from your handlers
import { signUp } from './SignupHandler.js';
import { signIn } from './SigninHandler.js';
import { signOut } from './SignoutHandler.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");

    // Elements for signup
    const userInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const passInput = document.getElementById('password');
    const signupBtn = document.getElementById('signup-btn');

    console.log(userInput, emailInput, passInput, signupBtn);

    // Elements for signin
    const emailSignInInput = document.getElementById('login-email');
    const passSignInInput = document.getElementById('login-password');
    const signinBtn = document.getElementById('login-btn');

    console.log(emailSignInInput, passSignInInput, signinBtn);

    // Element for signout
    const signoutBtn = document.getElementById('signout-btn');

    // Attach event listener for signup
    if (userInput && emailInput && passInput && signupBtn) {
        signupBtn.addEventListener('click', () => {
            const user = userInput.value.trim();
            const email = emailInput.value.trim();
            const password = passInput.value.trim();

            if (!user || !email || !password) {
                console.error("All fields are required for signup.");
                return;
            }

            signUp(user, email, password)
                .then(() => console.log("Signup successful"))
                .catch(error => console.error("Signup failed:", error.message));
        });
    }

    // Attach event listener for signin
    if (emailSignInInput && passSignInInput && signinBtn) {
        signinBtn.addEventListener('click', () => {
            const email = emailSignInInput.value.trim();
            const password = passSignInInput.value.trim();

            if (!email || !password) {
                console.error("All fields are required for signin.");
                return;
            }

            signIn(email, password)
                .then(() => console.log("Signin successful"))
                .catch(error => console.error("Signin failed:", error.message));
        });
    }

    // Attach event listener for signout
    if (signoutBtn) {
        signoutBtn.addEventListener('click', () => {
            signOut()
                .then(() => console.log("Signout successful"))
                .catch(error => console.error("Signout failed:", error.message));
        });
    }
});

