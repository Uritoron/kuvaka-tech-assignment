# src/api/v1/subscription.py
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
import stripe
from src.core.config import settings
from src.models import User # Only import User
from src.database.session import get_db
from src.core.security import get_current_user
import logging

router = APIRouter(prefix="/subscribe", tags=["subscription"])
logger = logging.getLogger(__name__)

# Configure Stripe SDK with your secret key
stripe.api_key = settings.STRIPE_SECRET_KEY

# --- POST /subscribe/pro ---
@router.post("/pro", status_code=status.HTTP_201_CREATED)
def create_pro_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Creates a Stripe Checkout Session for a Pro subscription.
    Returns the session URL for redirection.
    """
    try:
        # TODO: Replace 'price_...' with your actual Stripe Price ID for the Pro plan.
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price':'price_1S6ySI2LHczgbBNXFmM7ozWl', # <--- REPLACE WITH YOUR ACTUAL STRIPE PRICE ID
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=f"{settings.FRONTEND_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/cancel",
            #customer_email=current_user.mobile_number,
            client_reference_id=str(current_user.id),
            metadata={
                'user_id': str(current_user.id),
                'mobile_number': current_user.mobile_number
            }
        )
        logger.info(f"✅ Created Stripe Checkout Session for user {current_user.id}: {checkout_session.id}")
        return {"url": checkout_session.url}

    except Exception as e:
        logger.error(f"❌ Error creating Stripe Checkout Session for user {current_user.id}: {e}")
        # Return a more user-friendly error message if possible, but log the full error
        raise HTTPException(status_code=500, detail="Failed to create checkout session. Please try again later.")


# --- POST /webhook/stripe ---
@router.post("/webhook", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Webhook endpoint for Stripe to send event notifications.
    Handles subscription lifecycle events like payment success/failure.
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        # Verify the event was sent by Stripe
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
        
    except ValueError as e:
        # Invalid payload
        logger.warning(f"⚠️ Invalid Stripe webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logger.warning(f"⚠️ Invalid Stripe webhook signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # --- Handle the event ---
    logger.info(f"🔔 Received Stripe event: {event['type']} - ID: {event['id']}")
    
    # --- Successful Payment / Subscription Start ---
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Get user ID from session metadata
        user_id_str = session.get('metadata', {}).get('user_id') or session.get('client_reference_id')
        
        if user_id_str:
            try:
                user_id = int(user_id_str)
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    old_tier = user.subscription_tier
                    user.subscription_tier = "Pro"
                    # Consider resetting daily message count upon upgrade
                    # user.daily_message_count = 0 
                    # user.last_message_date = datetime.utcnow() # Reset date too if count is reset
                    db.commit()
                    logger.info(f"✅ User {user_id} upgraded from '{old_tier}' to 'Pro' tier via session {session['id']}")
                    
                    # Removed SubscriptionEvent logging
                    
                else:
                    logger.warning(f"⚠️ User with ID {user_id} not found for session {session['id']}")
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Error parsing user ID '{user_id_str}' from session {session['id']}: {e}")
        else:
            logger.warning(f"⚠️ No user ID found in session metadata/client_reference_id for session {session['id']}")

    # --- Failed Payment ---
    elif event['type'] == 'invoice.payment_failed':
        invoice = event['data']['object']
        logger.info(f"⚠️ Stripe Invoice Payment Failed: Invoice ID {invoice.get('id')}, Customer {invoice.get('customer')}")
        
        # Removed SubscriptionEvent logging
        
        # Optional: If you store Stripe Customer ID on your User model,
        # find the user and potentially downgrade them.
        # customer_id = invoice.get('customer')
        # user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        # if user and user.subscription_tier == "Pro":
        #     old_tier = user.subscription_tier
        #     user.subscription_tier = "Basic"
        #     db.commit()
        #     logger.info(f"⚠️ User {user.id} downgraded from '{old_tier}' to 'Basic' due to payment failure.")

    # --- Subscription Cancellation (Optional) ---
    elif event['type'] == 'customer.subscription.deleted':
         subscription = event['data']['object']
         logger.info(f"⚠️ Stripe Subscription Deleted: {subscription.get('id')}")
         # ... (potentially update user if linked via customer ID) ...
         
    # Acknowledge receipt
    return {"status": "received"}

# --- GET /subscription/status ---
@router.get("/status")
def get_subscription_status(
    current_user: User = Depends(get_current_user)
):
    """
    Returns the current subscription tier and usage information for the user.
    """
    daily_limit = 5 if current_user.subscription_tier == "Basic" else "Unlimited"
    
    return {
        "tier": current_user.subscription_tier,
        "daily_limit": daily_limit,
        "messages_used_today": current_user.daily_message_count,
        "last_reset_date": current_user.last_message_date.isoformat() if current_user.last_message_date else None
    }
