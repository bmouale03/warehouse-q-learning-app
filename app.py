from flask import Flask, render_template, request, redirect, url_for, send_file
from qlearning import route, best_route
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        start = request.form['start'].strip().upper()
        end = request.form['end'].strip().upper()
        intermediary = request.form.get('intermediary', '').strip().upper()

        # Validation des entrées
        if not (start in location_to_state and end in location_to_state):
            raise ValueError("Points de départ/arrivée invalides")
        
        if intermediary and intermediary not in location_to_state:
            raise ValueError("Point intermédiaire invalide")

        if intermediary:
            path = best_route(start, end, intermediary)
            route_type = "avec point intermédiaire"
        else:
            path = route(start, end)
            route_type = "directe"

        path_str = " → ".join(path)
        
        # Logging
        with open('routes_log.csv', 'a') as f:
            f.write(f"{datetime.now()},{start},{end},{intermediary if intermediary else 'None'},{path_str}\n")
        
        return render_template('results.html', 
                            path=path,
                            path_str=path_str,
                            start=start,
                            end=end,
                            intermediary=intermediary if intermediary else None,
                            route_type=route_type)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/download')
def download():
    return send_file('routes_log.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)