# app.py

from flask import Flask, request, jsonify, render_template
from functools import wraps
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from werkzeug.exceptions import BadRequest, HTTPException
from datetime import timedelta
import networkx as nx
import ssl
import os
import traceback
import logging
import json
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt, verify_jwt_in_request
)
from werkzeug.security import generate_password_hash, check_password_hash

# Import load_dotenv from python-dotenv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Retrieve environment variables
sqlUser = os.environ.get('SQL_USER')
sqlPass = os.environ.get('SQL_PASS')
sqlServer = os.environ.get('SQL_SERVER')
jwtSecretKey = os.environ.get('JWT_SECRET_KEY')
sqlDbName = os.environ.get('SQL_DATABASE')
localPath = os.environ.get('LOCAL_PATH')

# Verify that essential environment variables are set
if not all([sqlUser, sqlPass, sqlServer, jwtSecretKey]):
    raise RuntimeError("One or more environment variables are missing. Please check your .env file.")

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{sqlUser}:{sqlPass}@{sqlServer}/{sqlDbName}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = jwtSecretKey
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_ALGORITHM'] = 'HS256'

# Initialize extensions without passing 'app'
db = SQLAlchemy()
jwt = JWTManager()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_app(app):
    db.init_app(app)
    jwt.init_app(app)
    with app.app_context():
        db.create_all()

init_app(app)

# User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='general')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Global variables
bayes_net = None
inference_engine = None

# Helper functions
def validate_network_structure(nodes, edges):
    if not nodes or not edges:
        raise ValueError("Nodes and edges must be provided")
    
    G = nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph is not acyclic")
    
    edge_nodes = set([node for edge in edges for node in edge])
    if not edge_nodes.issubset(set(nodes)):
        raise ValueError("Edges contain nodes not present in the nodes list")

def validate_cpd(cpd_info):
    required_fields = ['variable', 'variable_card', 'values']
    for field in required_fields:
        if field not in cpd_info:
            raise ValueError(f"Missing required field: {field}")
        
    if cpd_info.get('evidence') and len(cpd_info['evidence']) != len(cpd_info['evidence_card']):
        raise ValueError("Length of evidence and evidence_card must match")

# Decorators
def role_required(required_role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt()
            user_role = claims.get('role')
            if user_role != required_role:
                return jsonify({'status': 'error', 'message': f'{required_role.capitalize()} access required'}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# Routes
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Welcome to the Bayesian Network API. Error: {str(e)}"

@app.route('/login', methods=['POST'])
def login():
    logger.info('Login route accessed')
    data = request.get_json() or request.form
    username = data.get('username')
    password = data.get('password')
    
    logger.info(f'Login attempt for username: {username}')
    
    if not username or not password:
        logger.info('Username or password not provided')
        return jsonify({'error': 'Username and password are required'}), 400

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        additional_claims = {'role': user.role}
        access_token = create_access_token(identity=user.id, additional_claims=additional_claims)
        logger.info(f'Authentication successful for username: {username}')
        return jsonify({'token': access_token}), 200
    logger.info(f'Invalid credentials for username: {username}')
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/upload_network', methods=['POST'])
@role_required('admin')
def upload_network():
    global bayes_net, inference_engine
    data = request.get_json()
    if not data:
        raise BadRequest("No data provided")

    nodes = data.get('nodes')
    edges = data.get('edges')
    cpds_data = data.get('cpds')

    try:
        validate_network_structure(nodes, edges)
        bayes_net = BayesianNetwork(edges)

        cpds = []
        for cpd_info in cpds_data:
            validate_cpd(cpd_info)
            cpd = TabularCPD(
                variable=cpd_info['variable'],
                variable_card=cpd_info['variable_card'],
                values=cpd_info['values'],
                evidence=cpd_info.get('evidence'),
                evidence_card=cpd_info.get('evidence_card')
            )
            cpds.append(cpd)
        bayes_net.add_cpds(*cpds)

        if not bayes_net.check_model():
            raise ValueError('Invalid Bayesian Network model')

        inference_engine = VariableElimination(bayes_net)

        return jsonify({'status': 'success', 'message': 'Bayesian Network uploaded successfully.'}), 200
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"An error occurred in upload_network: {str(e)}\n{tb}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred'}), 500

@app.route('/query', methods=['POST'])
@role_required('admin')
def query_model():
    global inference_engine
    if not inference_engine:
        return jsonify({'status': 'error', 'message': 'Inference engine not initialized.'}), 400

    data = request.get_json()
    if not data:
        raise BadRequest("No data provided")

    query_var = data.get('query_var')
    evidence = data.get('evidence', {})

    if not query_var:
        raise ValueError("No query variable provided")

    try:
        result = inference_engine.query(variables=[query_var], evidence=evidence)
        result_json = {query_var: result[query_var].values.tolist()}
        return jsonify(result_json), 200
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"An error occurred in query_model: {str(e)}\n{tb}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred'}), 500

@app.route('/get_network_structure', methods=['GET'])
@role_required('admin')
def get_network_structure():
    global bayes_net
    if not bayes_net:
        return jsonify({'status': 'error', 'message': 'No network structure available.'}), 400
    
    try:
        nodes = list(bayes_net.nodes())
        edges = list(bayes_net.edges())
        cpds = [cpd.to_dict() for cpd in bayes_net.get_cpds()]
        
        return jsonify({
            'status': 'success',
            'nodes': nodes,
            'edges': edges,
            'cpds': cpds
        }), 200
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"An error occurred in get_network_structure: {str(e)}\n{tb}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred'}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"status": "error", "message": str(e)}), 400

@app.errorhandler(ValueError)
def handle_value_error(e):
    return jsonify({"status": "error", "message": str(e)}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    tb = traceback.format_exc()
    logger.error(f"An error occurred: {str(e)}\n{tb}")
    return jsonify({
        "status": "error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    local_path = localPath
    cert_path = localPath + 'sims_cert_and_inter.crt'
    key_path = localPath + 'sims_private.key'
    
    if os.path.exists(cert_path) and os.path.exists(key_path):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(cert_path, key_path)
            logger.info("SSL certificate and key loaded successfully.")

            # Updated SSL configuration
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
            context.options = (
                ssl.OP_CIPHER_SERVER_PREFERENCE |      # Prefer server's cipher order
                ssl.OP_SINGLE_DH_USE |                 # Improve forward secrecy
                ssl.OP_SINGLE_ECDH_USE |               # Improve forward secrecy
                ssl.OP_NO_COMPRESSION                  # Disable TLS compression (CRIME attack)
            )

            # Explicitly disable TLS compression (in case OP_NO_COMPRESSION is not supported)
            context.options |= getattr(ssl, 'OP_NO_COMPRESSION', 0)

            logger.info("Starting Flask app with HTTPS...")
            app.run(host='0.0.0.0', port=8000, ssl_context=context)
        except ssl.SSLError as ssl_err:
            logger.error(f"SSL Error when loading certificate and key: {ssl_err}")
            logger.error("Error details:", exc_info=True)
            logger.error("Aborting due to SSL configuration error.")
            exit(1)
        except Exception as e:
            logger.error(f"Unexpected error when setting up SSL: {e}", exc_info=True)
            logger.error("Aborting due to unexpected SSL error.")
            exit(1)
    else:
        missing = []
        if not os.path.exists(cert_path):
            missing.append(f"Certificate file not found at {cert_path}")
        if not os.path.exists(key_path):
            missing.append(f"Private key file not found at {key_path}")
        logger.error("SSL certificate files missing:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("Aborting due to missing SSL certificates.")
        exit(1)