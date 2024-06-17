from django import forms

class PlotForm(forms.Form):
    x_text = forms.CharField(label='X Axis Label', max_length=100, initial='Machine Learning Engineer')
    y_text = forms.CharField(label='Y Axis Label', max_length=100, initial='Data Scientist')
    z_text = forms.CharField(label='Z Axis Label', max_length=100, initial='Accountant')