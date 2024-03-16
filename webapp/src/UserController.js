// UserController.js
import AuthHandler from './AuthHandler.js'; // Adjust the path as necessary

class UserController {
    constructor() {
        this.authHandler = new AuthHandler();
    }

    createUser(user, email, password) {
        this.authHandler.signUp(email, password)
            .then(user => {
                console.log("User created successfully", user);
                // Perform additional actions upon successful signup, if needed
            })
            .catch(error => {
                console.error("Error during user signup", error);
                // Handle signup errors (e.g., show an error message to the user)
            });
    }
    

    signinUser(email, password) {
        this.authHandler.signIn(email, password)
            .then(user => {
                console.log("User successfully signed in.", user);
                // Perform additional actions upon successful signup, if needed
            })
            .catch(error => {
                console.error("Error during user signup", error);
                // Handle signup errors (e.g., show an error message to the user)
            });
    }

    signoutUser() {
        this.authHandler.signOut()
            .then(user => {
                console.log("User successfully signed out.", user);
                // Perform additional actions upon successful signup, if needed
            })
            .catch(error => {
                console.error("Error during user sign out", error);
                // Handle signup errors (e.g., show an error message to the user)
            });
    }


}

export default UserController;

