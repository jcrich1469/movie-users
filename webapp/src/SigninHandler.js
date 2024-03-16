// Assuming this code is in your SignupHandler.js or wherever you handle form submissions
import UserController from './UserController.js'; // Adjust the path as necessary

const userController = new UserController();

export function signIn(email, password) {
    // Call createUser on your UserController instance
    userController.signinUser(email, password);
}
