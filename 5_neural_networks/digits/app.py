import sys, pygame as pg, numpy as np
from keras.models import load_model, Sequential

BLACK, WHITE, GRAY = (0,0,0), (255,255,255), (255,255,255)
ROWS, COLS, OFFSET, CELL = 28, 28, 20, 10

def validateArgs() -> str:
    """
    Validates command line arguments for the app.

    Checks if there is exactly one command line argument passed in,
    which should be the path to the model file. If not, exits the program
    with a usage message.

    Returns:
        str: The path to the model file
    """
    if len(sys.argv) != 2:
        sys.exit('App usage: python app.py path_to_model_file')
    return sys.argv[1]

def initGame() -> pg.Surface:
    """
    Initializes a pygame window of size 600x400 for the app.

    Returns:
        pg.Surface: The initialized pygame window
    """
    size = 600, 400
    pg.init()
    screen = pg.display.set_mode(size)
    pg.display.set_caption('Handwriting Classifcation')
    return screen

def loadFonts() -> tuple[pg.font.Font, pg.font.Font]:
    """
    Loads the OpenSans font from the font directory.

    Returns:
        tuple[pg.font.Font, pg.font.Font]: A tuple containing the small font (20px) and the large font (40px)
    """
    font_path = './font/OpenSans-Regular.ttf'

    try:
        small, large = pg.font.Font(font_path, 20), pg.font.Font(font_path, 40)
    except FileNotFoundError:
        print('')
        small, large = pg.font.SysFont(None, 20), pg.font.SysFont(None, 40)

    return small, large

def resetHandwriting() -> list[list[float]]:
    """
    Resets the handwriting data to an empty grid of zeros.

    Returns:
        list[list[float]]: A 2D list of floats representing the handwriting data
    """
    return [[0.0] * COLS for _ in range(ROWS)]

def createBtn(surface: pg.Surface, font: pg.font.Font, label: str, rect: pg.Rect, color: tuple=BLACK, bgColor: tuple=WHITE):
    """
    Creates a button on the given surface with the given label, rect, color and background color.

    Parameters:
        surface (pg.Surface): The surface to draw the button on
        font (pg.font.Font): The font to use for the button label
        label (str): The label to display on the button
        rect (pg.Rect): The rectangle representing the button's position and size
        color (tuple, optional): The color of the button label. Defaults to BLACK.
        bgColor (tuple, optional): The background color of the button. Defaults to WHITE.

    Returns:
        pg.Rect: The created button's rectangle
    """
    pg.draw.rect(surface, bgColor, rect)
    txt_surface = font.render(label, True, color)
    txt_rect = txt_surface.get_rect(center=rect.center)
    surface.blit(txt_surface, txt_rect)

    return rect

def drawGrid(surface: pg.Surface, handwriting: list[list[float]], mousePos: tuple[int, int]|None) -> None:
    """
    Draws a grid representing the handwriting data on the given surface.

    Parameters:
        surface (pg.Surface): The surface to draw the grid on
        handwriting (list[list[float]]): A 2D list of floats representing the handwriting data
        mousePos (tuple[int, int]|None): The current mouse position, or None if the mouse is not inside the grid

    Returns:
        None
    """
    # Iterate over each cell in the grid
    for i in range(ROWS):
        for j in range(COLS):
            # Calculate the position and size of the cell
            rect = pg.Rect(
                OFFSET + j * CELL,
                OFFSET + i * CELL,
                CELL, CELL
            )

            # Get the current value of the handwriting data at this cell
            cur_hwr = handwriting[i][j]

            # If the handwriting data is not zero at this cell, draw a grey rectangle
            if cur_hwr:
                # Calculate the shade of grey based on the handwriting value
                channel = 255 - int(cur_hwr * 255)
                clr = (channel, channel, channel)
            # Otherwise, draw a white rectangle
            else:
                clr = WHITE

            # Draw the rectangle
            pg.draw.rect(surface, clr, rect)

            # Draw a black border around the rectangle
            pg.draw.rect(surface, BLACK, rect, 1)

            # If the mouse is inside this cell, update the handwriting data
            if mousePos and rect.collidepoint(mousePos):
                # Set the value of the handwriting data to 0.98 (a light grey)
                handwriting[i][j] = 250/255

                # For each of the 3 neighbours of this cell, update their values to be 0.86 (a darker grey)
                for di, dj in [(1,0),(0,1),(1,1)]:
                    ni, nj = i+di,j+dj
                    if 0 <= ni < ROWS and 0<= nj < COLS:
                        handwriting[ni][nj] = max(handwriting[ni][nj], 220/255)

def main():
    """
    Main function of the program.

    This function validates the command line arguments, loads the model from
    the given path, initializes the game, loads the fonts, resets the
    handwriting data, creates the clock, and enters the main game loop.

    In the main game loop, it processes the events, redraws the grid based
    on the current state of the handwriting data, draws the reset and classify
    buttons, and updates the classification text based on the model's prediction.

    Exits the program when the user closes the window.
    """
    # Validate the command line arguments
    model_path = validateArgs()

    try:
        # Load the model from the given path
        model = load_model(model_path)
    except Exception as e:
        # Exit the program if an error occurs while loading the model
        sys.exit(f'Error while loading model. {e}')

    if not isinstance(model, Sequential):
        # Exit the program if the model is not a Sequential model
        sys.exit('Model is not a Sequential model')

    # Initialize the game
    screen = initGame()
    small_font, large_font = loadFonts()
    # Reset the handwriting data
    handwriting = resetHandwriting()
    # Create the clock
    clk = pg.time.Clock()
    # Initialize the classification text
    classification = None

    # Calculate the position and size of the grid
    grid_h = OFFSET * 2 + ROWS * CELL
    # Calculate the position and size of the reset button
    reset_btn_rect = pg.Rect(30, grid_h + 30, 100, 30)
    # Calculate the position and size of the classify button
    class_btn_rect = pg.Rect(150, grid_h + 30, 100, 30)

    # Flag to indicate whether the game is running or not
    running = True
    while running:
        # Get the current mouse position
        mouse_pos = None
        for evt in pg.event.get():
            etype = evt.type
            if etype == pg.QUIT:
                # Exit the game if the user closes the window
                running = False
            elif etype == pg.MOUSEBUTTONDOWN:
                # Update the current mouse position if the user clicks
                mouse_pos = evt.pos

        # Get the current mouse click status
        mouse_clicked = pg.mouse.get_pressed()[0]
        # Get the current mouse position if the user is clicking
        cur_mouse_pos = pg.mouse.get_pos() if mouse_clicked else None

        # Clear the screen
        screen.fill(BLACK)
        # Draw the grid based on the current state of the handwriting data
        drawGrid(screen, handwriting, cur_mouse_pos)

        # Draw the reset button
        reset_btn = createBtn(screen, small_font, 'Reset', reset_btn_rect)
        # Draw the classify button
        class_btn = createBtn(screen, small_font, 'Classify', class_btn_rect)

        # Check if the user has clicked on the reset button
        if mouse_pos:
            if reset_btn.collidepoint(mouse_pos):
                # Reset the handwriting data if the user clicks on the reset button
                handwriting = resetHandwriting()
                # Reset the classification text
                classification = None
            # Check if the user has clicked on the classify button
            elif class_btn.collidepoint(mouse_pos):
                try:
                    # Predict the classification of the handwriting data
                    inp = np.array(handwriting).reshape(1, 28, 28, 1)
                    pred = model.predict(inp, verbose='0')
                    # Update the classification text based on the model's prediction
                    classification = pred.argmax()
                except Exception as e:
                    print(f'Prediction error: {e}')
                    # Update the classification text to 'Error' if an error occurs
                    classification = 'Error'

        # Check if the classification text is not None
        if classification is not None:
            # Render the classification text using the large font
            class_txt = large_font.render(str(classification), True, WHITE)
            # Calculate the position of the classification text
            grid_w = OFFSET * 2 + CELL * COLS
            class_rect = class_txt.get_rect(
                center=(grid_w + (600 - grid_w) // 2, 100)
            )
            # Draw the classification text
            screen.blit(class_txt, class_rect)

        
        pg.display.flip() # Update the display
        clk.tick(60) # Limit the frame rate to 60 FPS

    # Exit the game
    pg.quit()
    sys.exit()


if __name__ == '__main__':
    main()