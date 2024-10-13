from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class BayesianNetwork(db.Model):
    __tablename__ = 'bayesian_networks'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    nodes = db.relationship('Node', back_populates='network', cascade='all, delete-orphan')
    edges = db.relationship('Edge', back_populates='network', cascade='all, delete-orphan')

class Node(db.Model):
    __tablename__ = 'nodes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    variable_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    network_id = db.Column(db.Integer, db.ForeignKey('bayesian_networks.id'), nullable=False)
    
    network = db.relationship('BayesianNetwork', back_populates='nodes')
    cpt = db.relationship('CPT', back_populates='node', uselist=False, cascade='all, delete-orphan')

class Edge(db.Model):
    __tablename__ = 'edges'
    id = db.Column(db.Integer, primary_key=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('nodes.id'), nullable=False)
    child_id = db.Column(db.Integer, db.ForeignKey('nodes.id'), nullable=False)
    network_id = db.Column(db.Integer, db.ForeignKey('bayesian_networks.id'), nullable=False)
    
    network = db.relationship('BayesianNetwork', back_populates='edges')
    parent = db.relationship('Node', foreign_keys=[parent_id])
    child = db.relationship('Node', foreign_keys=[child_id])

class CPT(db.Model):
    __tablename__ = 'cpts'
    id = db.Column(db.Integer, primary_key=True)
    node_id = db.Column(db.Integer, db.ForeignKey('nodes.id'), nullable=False)
    data = db.Column(db.JSON, nullable=False)
    
    node = db.relationship('Node', back_populates='cpt')