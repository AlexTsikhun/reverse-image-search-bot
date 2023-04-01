import webbrowser

# book_name = "20266    Books v. Cigarettes"

def go_to_site(book_name):
    """
        Function, that open site with book_name

        Input:
            book_name - str (Name of book)
        Output:
            cite page with appropriate book
    """
    
    # Replace the URL with the book site you want to search
    url = "https://www.bookdepository.com/search?searchTerm=" + book_name.replace(" ", "+") + "&search=Find+book"
    # print(url)
    webbrowser.open_new_tab(url)
