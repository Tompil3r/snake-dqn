import PySimpleGUI as sg

# 166, 217, 72 / 142, 204, 58 / #74d948 / #8ecc3a
#sg.Window(title="Snake", layout=[[]], margins=(100, 50)).read()



class GUI(sg.Window):
    def __init__(self):
        self.width = 680 # 40 * 17
        self.height = 650 # 40 * 15

        self.board_width = 680
        self.board_height = 600
        
        self.score_width = self.width
        self.score_height = 50

        self.background_color = '#4a752c'
        self.board_colors = ['#74d948', '#8ecc3a']
        self.init_layout()

        super().__init__(title='Snake', layout=[[self.score_label], [self.canvas]], size=(self.width, self.height), margins=(0, 0), background_color=self.background_color)
        self.read()

    
    def init_variables():
        pass


    def init_layout(self):
        self.score_label = sg.Text('Best Score: 0', size=(20, 1), font=('Arial', 16), background_color=self.background_color)
        print(self.score_label.BackgroundColor)

        self.canvas = sg.Graph(
            canvas_size=(self.board_width, self.board_height),
            graph_bottom_left=(0, self.board_height),
            graph_top_right=(self.board_width, self.score_height),
            key="graph"
        )

    
    def draw_background():
        pass
'''


window = sg.Window(title='Snake', layout=[[canvas], [label]], size=(800, 1000), margins=(0, 0))
window.finalize()

draw_background(canvas)
# canvas.draw_rectangle((0, 0), (100, 500), fill_color='#74d948', line_width=0)
window.Read()
'''
w = GUI()