import uuid
from app.langgraph_backend import register_user, authenticate_user, retrieve_all_threads, save_thread_title

def run_tests():
    try:
        print("--- Testing Registration ---")
        email = f"test_{uuid.uuid4()}@example.com"
        res1 = register_user("TestUser", email, "password123")
        print("Register Output:", res1)
        assert res1["success"] == True, "Registration failed"

        print("\n--- Testing Authentication ---")
        res2 = authenticate_user(email, "password123")
        print("Authenticate Output:", res2)
        assert res2["success"] == True, "Authentication failed"
        user_id = res2["user_id"]

        print("\n--- Testing Invalid Authentication ---")
        res3 = authenticate_user(email, "wrongpassword")
        print("Invalid Auth Output:", res3)
        assert res3["success"] == False, "Invalid auth should fail"

        print("\n--- Testing Thread Title Saving ---")
        thread_id = str(uuid.uuid4())
        save_thread_title(thread_id, "Explain quantum computing in simple terms.", user_id)
        print(f"Thread {thread_id} saved successfully for user {user_id}")

        print("\n--- Testing Retrieve User Threads ---")
        threads = retrieve_all_threads(user_id)
        print("User Threads retrieved:", threads)
        assert any(t["id"] == thread_id for t in threads), "Thread not found in retrieval"

        print("\n✅ All self-tests passed successfully! The database schema and auth logic are working perfectly.")
    
    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")

if __name__ == "__main__":
    run_tests()
