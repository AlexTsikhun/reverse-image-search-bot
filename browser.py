import webbrowser


def go_to_site(book_name):
    """
        Function, that open site with book_name

        book_name = "20266    Books v. Cigarettes"

        Input:
            book_name - str (Name of book)
        Output:
            cite page with appropriate book
    """

    url = ("https://www.bookdepository.com/search?searchTerm=" +
           book_name.replace(" ", "+") + "&search=Find+book")
    webbrowser.open_new_tab(url)
