from django import forms


class UploadFileForm(forms.Form):
    model = forms.FileField()
    data = forms.FileField()
