import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from "firebase/auth";

// TODO: Replace with your actual Firebase config from GCP Console
const firebaseConfig = {
  apiKey: "AIzaSyAT_8B_luzmcgVzggLRKt0uPP-tFfCw-Gs",
  authDomain: "project-9f6fdf50-be3b-4371-a96.firebaseapp.com",
  projectId: "project-9f6fdf50-be3b-4371-a96",
  storageBucket: "project-9f6fdf50-be3b-4371-a96.firebasestorage.app",
  messagingSenderId: "355876710746",
  appId: "1:355876710746:web:5b1ca31aca2e770a63eb63",
  measurementId: "G-9R89CJKYNY"
};

// Initialize Firebase only if config is somewhat valid (prevents crash on default)
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

export const loginWithGoogle = async () => {
    try {
        const result = await signInWithPopup(auth, googleProvider);
        return result.user;
    } catch (error) {
        console.error("Error signing in with Google", error);
        throw error;
    }
};

export const logoutUser = async () => {
    try {
        await signOut(auth);
    } catch (error) {
        console.error("Error signing out", error);
        throw error;
    }
};
