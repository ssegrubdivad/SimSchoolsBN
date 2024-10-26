# app12.py

from flask import Flask, request, jsonify, render_template, send_file
from functools import wraps
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from werkzeug.exceptions import BadRequest, HTTPException
from werkzeug.utils import secure_filename
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

# Import new parsing modules
from src.input_parsing import NetworkParser, CPTParser
from src.query_interface.query_processor import QueryProcessor
from src.visualization.network_visualizer import NetworkVisualizer

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
app.config['UPLOAD_FOLDER'] = 'uploads/'  # New configuration for file uploads

# Initialize extensions without passing 'app'
db = SQLAlchemy()
jwt = JWTManager()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_app(app):
    db.init_app(app)
    jwt.init_app(app)
    with app.app_context():
        db.create_all()

init_app(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
query_processor = None

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
    global bayes_net, query_processor
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if file and file.filename.endswith('.bns'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            parser = NetworkParser()
            bayes_net = parser.parse_file(filepath)
            query_processor = QueryProcessor(bayes_net)
            
            # Check for validation errors
            if hasattr(query_processor, 'validation_errors'):
                logger.warning(f"Network uploaded but has validation errors: {query_processor.validation_errors['message']}")
                return jsonify({
                    "status": "warning",
                    "message": "Network structure uploaded but has validation issues",
                    "validation_errors": query_processor.validation_errors['message'],
                    "nodes": len(bayes_net.nodes),
                    "edges": len(bayes_net.edges)
                }), 200
            
            logger.info(f"Network uploaded successfully: {len(bayes_net.nodes)} nodes, {len(bayes_net.edges)} edges")
            return jsonify({
                "status": "success",
                "message": "Network uploaded and parsed successfully",
                "nodes": len(bayes_net.nodes),
                "edges": len(bayes_net.edges)
            }), 200
        except ValueError as ve:
            logger.warning(f"Invalid network file: {str(ve)}")
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            logger.error(f"Error parsing network file: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error", 
                "message": f"An unexpected error occurred while parsing the network file: {str(e)}"
            }), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type. Please upload a .bns file"}), 400

@app.route('/upload_cpt', methods=['POST'])
@role_required('admin')
def upload_cpt():
    global bayes_net, query_processor
    if not bayes_net:
        return jsonify({
            "status": "error",
            "message": "No network structure available. Please upload a network first."
        }), 400
    
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file was uploaded. Please select a CPT file to upload."
        }), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file was selected. Please choose a CPT file to upload."
        }), 400
    if file and file.filename.endswith('.cpt'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            parser = CPTParser(bayes_net)
            cpts = parser.parse_file(str(filepath))
            
            # If we reach this point, all CPTs were parsed successfully
            for node_id, cpt in cpts.items():
                existing_cpd = bayes_net.get_cpds(node_id)
                if existing_cpd:
                    bayes_net.remove_cpds(node_id)
                bayes_net.add_cpds(cpt)

            # Reinitialize the query processor with the updated Bayesian Network
            query_processor = QueryProcessor(bayes_net)
            
            # Check for validation errors
            if hasattr(query_processor, 'validation_errors'):
                return jsonify({
                    "status": "warning",
                    "message": "CPTs uploaded but there are validation issues",
                    "validation_errors": query_processor.validation_errors['message'],
                    "uploaded_cpts": list(cpts.keys())
                }), 200

            return jsonify({
                "status": "success",
                "message": f"All {len(cpts)} CPTs uploaded and validated successfully.",
                "uploaded_cpts": list(cpts.keys())
            }), 200
        except ValueError as e:
            error_message = f"Error in CPT file: {str(e)}"
            logger.error(error_message)
            return jsonify({
                "status": "error", 
                "message": "We encountered an error in your CPT file.",
                "details": str(e)
            }), 400
        except KeyError as e:
            error_message = f"Missing key in CPT file: {str(e)}"
            logger.error(error_message)
            return jsonify({
                "status": "error", 
                "message": "Error in CPT file structure. Missing key information.",
                "details": f"Missing information for: {str(e)}"
            }), 400
        except Exception as e:
            error_message = f"An unexpected error occurred while processing the CPT file: {str(e)}"
            logger.error(error_message, exc_info=True)
            return jsonify({
                "status": "error", 
                "message": "An unexpected error occurred. Please check the server logs for details.",
                "details": str(e)
            }), 500
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Please upload a .cpt file."
        }), 400
        
@app.route('/query', methods=['POST'])
@role_required('admin')
def query_model():
    global query_processor
    if not query_processor:
        return jsonify({
            'status': 'error',
            'message': 'Query processor not initialized. Please upload a network first.'
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'No data provided'
        }), 400

    query_type = data.get('query_type', 'marginal')
    inference_algorithm = data.get('inference_algorithm', 'variable_elimination')
    query_vars = data.get('query_vars', [])
    evidence = data.get('evidence', {})
    interventions = data.get('interventions', {})
    time_steps = data.get('time_steps', None)

    if not query_vars and query_type != 'mpe':
        return jsonify({
            'status': 'error',
            'message': 'No query variables provided'
        }), 400

    try:
        query_processor.set_inference_algorithm(inference_algorithm)
        
        if query_type == 'temporal':
            if time_steps is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Time steps must be provided for temporal queries'
                }), 400
            result = query_processor.temporal_query(query_vars, time_steps, evidence)
        else:
            result = query_processor.process_query(query_type, query_vars, evidence, interventions)
        
        return jsonify({'status': 'success', 'result': result}), 200
        
    except ValueError as ve:
        # Handle validation errors specifically
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'validation_error'
        }), 400
    except Exception as e:
        logger.error(f"An error occurred in query_model: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred while processing the query. Please check that all probability distributions are properly specified and supported.',
            'error_type': 'processing_error'
        }), 500

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

@app.route('/visualize_network', methods=['GET'])
@role_required('admin')
def visualize_network():
    global bayes_net
    if not bayes_net:
        return jsonify({'status': 'error', 'message': 'No network structure available.'}), 400
    
    try:
        visualizer = NetworkVisualizer(bayes_net)
        graph_data = visualizer.generate_html()
        return graph_data, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"An error occurred in visualize_network: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': 'An unexpected error occurred while generating the visualization.',
            'details': str(e)
        }), 500

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