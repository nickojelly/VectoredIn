# Django Search Application

This is a Django-based web application that provides search functionality using OpenAI's language model and the HNSW (Hierarchical Navigable Small World) algorithm for efficient similarity search.

## Features

- Search functionality powered by OpenAI's language model
- Efficient similarity search using the HNSW algorithm
- User-friendly web interface for performing searches
- Admin panel for managing the application's data and settings

## Installation

1. Clone the repository:



git clone https://github.com/nickojelly/JobMarketAnalyzer


2. Navigate to the project directory:



cd django-search-app


3. Create and activate a virtual environment:



python -m venv venv source venv/bin/activate # For Unix/Linux venv\Scripts\activate # For Windows


4. Install the required dependencies:



pip install -r requirements.txt


5. Apply database migrations:



python manage.py migrate


6. Start the development server:



python manage.py runserver


7. Access the application in your web browser at `http://localhost:8000`.

## Configuration

- Update the `settings.py` file to configure the application's settings, such as database connection, secret key, and allowed hosts.
- Customize the templates in the `templates` directory to match your desired design and layout.
- Modify the `urls.py` files to define the URL patterns for your application's views.

## Usage

- Use the web interface to perform searches by entering query terms.
- The application will utilize OpenAI's language model to understand the query and retrieve relevant results.
- The HNSW algorithm will be used to efficiently find similar items based on the query.
- Explore the admin panel to manage the application's data and settings.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).



Feel free to customize and expand upon this description based on the specific details and features of your Django search application.
