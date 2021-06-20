from django import forms


class UploadFileForm(forms.Form):
    model = forms.FileField()
    data = forms.FileField()


class PredictForm(forms.Form):

    def __init__(self, *args, **kwargs):
        features = kwargs.pop('features')
        super(PredictForm, self).__init__(*args, **kwargs)

        for i, feature in enumerate(features):
            self.fields['feature_%s' % i] = forms.CharField(label=feature)
