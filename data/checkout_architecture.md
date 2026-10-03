# Checkout Service Architecture

The checkout service is responsible for creating customer orders.

The service runs on AWS ECS.

Checkout communicates with:

- PostgreSQL
- Redis
- Payment Service

Traffic enters through an AWS Application Load Balancer.

Container images are stored in Amazon ECR.