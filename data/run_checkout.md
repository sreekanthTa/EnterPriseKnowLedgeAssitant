# Checkout Service - HTTP 500 Runbook

The checkout service can return HTTP 500 errors when
its dependencies are unavailable.

## Investigation Steps

1. Check ECS task health.
2. Check application logs.
3. Check PostgreSQL connectivity.
4. Check Redis connectivity.
5. Check the payment service.
6. Check recent deployments.

If the errors started immediately after a deployment,
compare the current deployment with the previous version.

## Common Causes

Common causes include:

- PostgreSQL connection exhaustion
- Redis unavailable
- Payment service failure
- Application deployment problems