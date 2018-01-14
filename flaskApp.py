import flask
import sys
from flask import render_template

app = flask.Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def welcomePage():
    return render_template('welcome.html')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {0} host port'.format(sys.argv[0]), file=sys.stderr)
        exit()

    host = sys.argv[1]
    port = sys.argv[2]
    app.run(host=host, port=int(port))
