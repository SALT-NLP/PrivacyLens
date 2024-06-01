"""Virtual tools created from json files."""
from toolemu.tools.tool_interface import (
    ArgException,
    ArgParameter,
    ArgReturn,
    FunctionToolkit,
    VirtualFunctionTool,
)
from toolemu.utils.my_typing import *

from .register import register_toolkit


#################### GoogleCalendar ####################


class GoogleCalendarReadEvents(VirtualFunctionTool):
    name = "GoogleCalendarReadEvents"
    summary = "Read events by event_ids."
    parameters: List[ArgParameter] = [
        {
            "name": "event_ids",
            "type": "array",
            "description": "An array of unique event identifiers to read events.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "event_details",
            "type": "array",
            "description": "An array of objects, each containing the details of an event, including event_id, event_name, content, start_time, end_time, location, and attendees.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "NotFoundException",
            "description": "One or multiple 'event_ids' do not exist.",
        }
    ]


class GoogleCalendarSearchEvents(VirtualFunctionTool):
    name = "GoogleCalendarSearchEvents"
    summary = "Search events by keywords, date range, or attendees. If certain arguments are not provided, the corresponding filters are not applied."
    parameters: List[ArgParameter] = [
        {
            "name": "keywords",
            "type": "array",
            "description": "A list of keywords to search for.",
            "required": False,
        },
        {
            "name": "start_date",
            "type": "string",
            "description": "The start date of the date range in ISO 8601 format.",
            "required": False,
        },
        {
            "name": "end_date",
            "type": "string",
            "description": "The end date of the date range in ISO 8601 format.",
            "required": False,
        },
        {
            "name": "attendees",
            "type": "array",
            "description": "An array of email addresses of the attendees to search for.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "events",
            "type": "array",
            "description": "An array of the searched event ids.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The start_date and end_date are not formatted correctly, or start_date is later than the end_date.",
        }
    ]


class GoogleCalendarSendInvitations(VirtualFunctionTool):
    name = "GoogleCalendarSendInvitations"
    summary = "Send event invitations to attendees."
    parameters: List[ArgParameter] = [
        {
            "name": "event_id",
            "type": "string",
            "description": "The unique identifier of the event.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "Indicates whether the invitations were successfully sent.",
        }
    ]
    exceptions: List[ArgException] = [
        {"name": "NotFoundException", "description": "The event_id does not exist."}
    ]


@register_toolkit()
class GoogleCalendar(FunctionToolkit):
    name_for_human = "Google Calendar management tool"
    description_for_human = "Tool for managing Google Calendar events."
    name_for_model = "GoogleCalendar"
    description_for_model = "Tool for managing Google Calendar events, including searching and sharing the calendar with other users."
    tool_classes = [
        GoogleCalendarReadEvents,
        GoogleCalendarSearchEvents,
        GoogleCalendarSendInvitations,
    ]


#################### Gmail ####################


class GmailSendEmail(VirtualFunctionTool):
    name = "GmailSendEmail"
    summary = "Send an email to one or multiple recipients, with optional scheduling, cc, and bcc."
    parameters: List[ArgParameter] = [
        {
            "name": "to",
            "type": "string",
            "description": "A string of email addresses to send to, separated by commas.",
            "required": True,
        },
        {
            "name": "subject",
            "type": "string",
            "description": "The subject of the email.",
            "required": True,
        },
        {
            "name": "body",
            "type": "string",
            "description": "The body of the email.",
            "required": True,
        },
        {
            "name": "cc",
            "type": "string",
            "description": "A string of email addresses to carbon copy (CC) to, separated by commas.",
            "required": False,
        },
        {
            "name": "bcc",
            "type": "string",
            "description": "A string of email addresses to blind carbon copy (BCC) to, separated by commas.",
            "required": False,
        },
        {
            "name": "send_at",
            "type": "string",
            "description": "The scheduled date and time to send the email in the format 'YYYY-MM-DD HH:mm'.",
            "required": False,
        },
        {
            "name": "attachments",
            "type": "array",
            "description": "An array of local file paths of attachments.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "An boolean value indicating whether the email was sent successfully.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The provided email address in 'to', 'cc', or 'bcc' is malformed, any file path in 'attachments' is invalid, or the 'send_at' is not in the correct format.",
        }
    ]


class GmailSearchEmails(VirtualFunctionTool):
    name = "GmailSearchEmails"
    summary = "Search for emails based on keywords, folders, labels, date range, or sender and recipient. If certain arguments are not provided, the corresponding filters are not applied."
    parameters: List[ArgParameter] = [
        {
            "name": "keywords",
            "type": "array",
            "description": "The list of keywords to search for.",
            "required": False,
        },
        {
            "name": "folders",
            "type": "array",
            "description": "The list of folders to search for. Each value should be a valid folder such as 'inbox', 'sent', 'drafts', or 'spam'.",
            "required": False,
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "The maximum number of emails to retrieve. Default is 5.",
            "required": False,
        },
        {
            "name": "date_range",
            "type": "object",
            "description": "An object containing 'start_date' and 'end_date' in the format 'YYYY-MM-DD' to filter emails by date range.",
            "required": False,
        },
        {
            "name": "from",
            "type": "string",
            "description": "The sender's email address to filter emails by.",
            "required": False,
        },
        {
            "name": "to",
            "type": "string",
            "description": "The recipient's email address to filter emails by.",
            "required": False,
        },
        {
            "name": "labels",
            "type": "array",
            "description": "The list of labels to filter emails by, e.g. 'important'.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "emails",
            "type": "array",
            "description": "An array of at most 'limit' emails that match the search criteria, each containing the 'id', 'subject', 'from', 'to', 'timestamp' (in the format 'YYYY-MM-DD HH:mm').",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'limit' is negative or not an integer, the 'date_range' has incorrect date format, or the 'from' or 'to' email addresses are invalid.",
        }
    ]


class GmailReadEmail(VirtualFunctionTool):
    name = "GmailReadEmail"
    summary = "Read the content of an email, including attachments."
    parameters: List[ArgParameter] = [
        {
            "name": "email_id",
            "type": "string",
            "description": "The unique identifier of the email.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "from",
            "type": "string",
            "description": "The email address the email was sent from.",
        },
        {
            "name": "to",
            "type": "string",
            "description": "A string of email addresses that the email was sent to, separated by commas.",
        },
        {
            "name": "cc",
            "type": "string",
            "description": "A string of email addresses that the email was cc'ed to, separated by commas.",
        },
        {
            "name": "bcc",
            "type": "string",
            "description": "A string of email addresses that the email was bcc'ed to, separated by commas.",
        },
        {
            "name": "subject",
            "type": "string",
            "description": "The subject of the email.",
        },
        {"name": "body", "type": "string", "description": "The body of the email."},
        {
            "name": "timestamp",
            "type": "string",
            "description": "The timestamp of when the email was sent (YYYY-MM-DD hh:mm:ss).",
        },
        {
            "name": "folder",
            "type": "string",
            "description": "The folder the email is in, such as 'inbox', 'sent', 'drafts', or 'spam'.",
        },
        {
            "name": "labels",
            "type": "array",
            "description": "The labels of the email (e.g., `important`).",
        },
        {
            "name": "attachments",
            "type": "array",
            "description": "An array of objects with attachment names and ids (attachment_id).",
        },
    ]
    exceptions: List[ArgException] = [
        {"name": "NotFoundException", "description": "The 'email_id' is not found."}
    ]


class GmailSearchContacts(VirtualFunctionTool):
    name = "GmailSearchContacts"
    summary = "Search for contacts based on filters. If certain arguments are not provided, the corresponding filters are not applied."
    parameters: List[ArgParameter] = [
        {
            "name": "name",
            "type": "string",
            "description": "The name of the contact to search for.",
            "required": False,
        },
        {
            "name": "email",
            "type": "string",
            "description": "The email of the contact to search for.",
            "required": False,
        },
        {
            "name": "remark",
            "type": "string",
            "description": "The remark of the contact to search for.",
            "required": False,
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "The maximum number of search results to return. Default is 5.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "contacts",
            "type": "array",
            "description": "An array of contact objects containing contact information, each object includes 'contact_id', 'name', 'email', 'remark', and 'info'.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'limit' is negative or not an integer.",
        }
    ]


@register_toolkit()
class Gmail(FunctionToolkit):
    name_for_human = "Gmail"
    description_for_human = "Tool for managing emails and contacts in Gmail."
    name_for_model = "Gmail"
    description_for_model = "Tool for managing emails and contacts in Gmail. Use it to send, search, read emails, as well as search contacts."
    tool_classes = [
        GmailSendEmail,
        GmailSearchEmails,
        GmailReadEmail,
        GmailSearchContacts,
    ]


#################### Slack ####################


class SlackSendMessage(VirtualFunctionTool):
    name = "SlackSendMessage"
    summary = "Send a message and optional file to a channel or user."
    parameters: List[ArgParameter] = [
        {
            "name": "recipient",
            "type": "string",
            "description": "The name of the recipient, channel name starts with '#', e.g., '#general', or user name starts with '@', e.g., '@john'.",
            "required": True,
        },
        {
            "name": "message",
            "type": "string",
            "description": "The content of the message.",
            "required": True,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "The local file path of the file to upload (optional).",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "Whether the message was sent successfully.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'recipient' is in an invalid format, or the 'message' is empty.",
        },
        {
            "name": "NotFoundException",
            "description": "The 'recipient' does not exist, or the 'file_path' does not exist.",
        },
    ]


class SlackSearchMessage(VirtualFunctionTool):
    name = "SlackSearchMessage"
    summary = "Search for messages in a channel or direct messages."
    parameters: List[ArgParameter] = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query.",
            "required": True,
        },
        {
            "name": "in",
            "type": "string",
            "description": "The channel name or user name to search within, must start with '#' (for channels) or '@' (for users). When not specified, search all channels and direct messages.",
            "required": False,
        },
        {
            "name": "from",
            "type": "string",
            "description": "The user name to search for, must start with '@'. When not specified, search all users.",
            "required": False,
        },
        {
            "name": "max_results",
            "type": "integer",
            "description": "The maximum number of results to return. Default is 10.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "messages",
            "type": "array",
            "description": "The search results, each object includes 'content' (string, the content of the message), 'timestamp' (string, the timestamp of the message in ISO 8601 format), 'in' (string, the channel name or user name), 'from' (string, the user name of the sender), and 'file_id' (string, the unique identifier of the file if the message is a file).",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'in' or 'from' argument is not a valid string or the 'max_results' argument is not a valid integer.",
        }
    ]


class SlackSearchChannelOrUser(VirtualFunctionTool):
    name = "SlackSearchChannelOrUser"
    summary = "Search for channels or users by query."
    parameters: List[ArgParameter] = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query.",
            "required": True,
        },
        {
            "name": "search_type",
            "type": "string",
            "description": "One of ['channels', 'users'], the type of resources to search.",
            "required": True,
        },
        {
            "name": "max_results",
            "type": "integer",
            "description": "The maximum number of results to return. Default is 10.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "results",
            "type": "array",
            "description": "A list of objects indicating the search results, each object includes 'name' (string, the name of the channel or user), 'status' (string, one of ['member', 'non-member', 'owner'] for channels, 'online' or 'offline' for users).",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'search_type' is not one of ['channels', 'users'], the 'max_results' is not a positive integer.",
        }
    ]


class SlackGetUserDetails(VirtualFunctionTool):
    name = "SlackGetUserDetails"
    summary = "Retrieve the profile and status of a user."
    parameters: List[ArgParameter] = [
        {
            "name": "user_name",
            "type": "string",
            "description": "The name of the user, must start with '@'.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "profile",
            "type": "object",
            "description": "The profile of the user, includes 'first_name' (string), 'last_name' (string), 'email' (string), 'phone' (string), 'local_time' (string), 'title' (string).",
        },
        {
            "name": "status",
            "type": "object",
            "description": "The status of the user, includes 'status_text' (string), 'status_emoji' (string), 'presence' (string).",
        },
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'user_name' is empty or in an invalid format.",
        }
    ]


@register_toolkit()
class Slack(FunctionToolkit):
    name_for_human = "Slack"
    description_for_human = "Toolkit for managing Slack interactions."
    name_for_model = "Slack"
    description_for_model = "A toolkit for managing Slack interactions, including sending messages, managing channels, searching for messages, channels, and users."
    tool_classes = [
        SlackSendMessage,
        SlackSearchMessage,
        SlackSearchChannelOrUser,
        SlackGetUserDetails,
    ]


#################### FacebookManager ####################


class FacebookManagerGetUserProfile(VirtualFunctionTool):
    name = "FacebookManagerGetUserProfile"
    summary = "Get the user's profile information."
    parameters: List[ArgParameter] = []
    returns: List[ArgReturn] = [
        {
            "name": "profile",
            "type": "object",
            "description": "The user's profile information containing fields such as user_id, name, email, birthday, and location.",
        }
    ]
    exceptions: List[ArgException] = []


class FacebookManagerUpdateUserProfile(VirtualFunctionTool):
    name = "FacebookManagerUpdateUserProfile"
    summary = "Update the user's profile information."
    parameters: List[ArgParameter] = [
        {
            "name": "updates",
            "type": "object",
            "description": "The profile information to update, containing fields such as name, email, birthday, and location.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "Whether the update was successful.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'updates' parameter contains invalid fields or values.",
        }
    ]


class FacebookManagerSearchFriends(VirtualFunctionTool):
    name = "FacebookManagerSearchFriends"
    summary = "Search for the user's friends by keyword."
    parameters: List[ArgParameter] = [
        {
            "name": "keyword",
            "type": "string",
            "description": "The keyword to search for.",
            "required": True,
        },
        {
            "name": "max_results",
            "type": "integer",
            "description": "The maximum number of search results to return. Default value is 10.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "friends_info",
            "type": "array",
            "description": "The search results containing objects with friend's user_id, name, email, birthday, and location.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'keyword' parameter is empty.",
        }
    ]


class FacebookManagerCreatePost(VirtualFunctionTool):
    name = "FacebookManagerCreatePost"
    summary = "Create a new post on the user's timeline."
    parameters: List[ArgParameter] = [
        {
            "name": "content",
            "type": "string",
            "description": "The content of the post.",
            "required": True,
        },
        {
            "name": "media_path",
            "type": "string",
            "description": "The local path of the media file to upload.",
            "required": False,
        },
        {
            "name": "privacy_setting",
            "type": "string",
            "description": "One of ['public', 'friends', 'only me'], the privacy setting for the post. Default value is 'friends'.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "post_id",
            "type": "string",
            "description": "The unique identifier of the created post.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'content' parameter is empty or the 'privacy_setting' parameter is not one of ['public', 'friends', 'only me'].",
        }
    ]


class FacebookManagerSearchPosts(VirtualFunctionTool):
    name = "FacebookManagerSearchPosts"
    summary = "Search for the user's own posts or other's posts by keyword."
    parameters: List[ArgParameter] = [
        {
            "name": "user_id",
            "type": "string",
            "description": "The unique identifier of the user whose posts to search for.",
            "required": True,
        },
        {
            "name": "keyword",
            "type": "string",
            "description": "The keyword to search for.",
            "required": True,
        },
        {
            "name": "max_results",
            "type": "integer",
            "description": "The maximum number of search results to return. Default value is 10.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "posts",
            "type": "array",
            "description": "The search results containing objects with post_id and content.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "NotFoundException",
            "description": "The 'user_id' parameter is not found.",
        },
        {
            "name": "InvalidRequestException",
            "description": "The 'keyword' parameter is empty.",
        },
    ]


@register_toolkit()
class FacebookManager(FunctionToolkit):
    name_for_human = "Facebook Manager"
    description_for_human = "Toolkit for managing a user's Facebook account."
    name_for_model = "FacebookManager"
    description_for_model = "FacebookManager is a toolkit for managing a user's Facebook account, including profile information, posts, comments, messages, and searching for friends. It also allows users to search their own posts or other's posts by keyword. The toolkit provides functionalities for creating, reading, posts and comments, sending messages, and searching for friends and posts."
    tool_classes = [
        FacebookManagerGetUserProfile,
        FacebookManagerUpdateUserProfile,
        FacebookManagerSearchFriends,
        FacebookManagerCreatePost,
        FacebookManagerSearchPosts,
    ]


#################### NotionManager ####################


class NotionManagerCreatePage(VirtualFunctionTool):
    name = "NotionManagerCreatePage"
    summary = "Creates a new page in Notion."
    parameters: List[ArgParameter] = [
        {
            "name": "page_title",
            "type": "string",
            "description": "The title of the new page.",
            "required": True,
        },
        {
            "name": "page_content",
            "type": "string",
            "description": "The content of the new page.",
            "required": True,
        },
        {
            "name": "parent_page_id",
            "type": "string",
            "description": "The id of the parent page. If not provided, the new page will be created at the root level.",
            "required": False,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "page_id",
            "type": "string",
            "description": "The id of the created page.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'page_title' or 'page_content' parameter is empty, or the 'parent_page_id' parameter is invalid.",
        },
        {
            "name": "NotFoundException",
            "description": "The specified parent page does not exist.",
        },
    ]


class NotionManagerReadPage(VirtualFunctionTool):
    name = "NotionManagerReadPage"
    summary = "Read the content of a page in Notion."
    parameters: List[ArgParameter] = [
        {
            "name": "page_id",
            "type": "string",
            "description": "The id of the page.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "page_content",
            "type": "string",
            "description": "The content of the page.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "NotFoundException",
            "description": "The specified page does not exist.",
        }
    ]


class NotionManagerEditPage(VirtualFunctionTool):
    name = "NotionManagerEditPage"
    summary = "Edits an existing page in Notion."
    parameters: List[ArgParameter] = [
        {
            "name": "page_id",
            "type": "string",
            "description": "The id of the page to be edited.",
            "required": True,
        },
        {
            "name": "new_content",
            "type": "string",
            "description": "The new content for the page.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "Whether the operation was successful.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'page_id' parameter is invalid, or the 'new_content' parameter is empty.",
        },
        {
            "name": "NotFoundException",
            "description": "The specified page does not exist.",
        },
    ]


class NotionManagerSharePage(VirtualFunctionTool):
    name = "NotionManagerSharePage"
    summary = "Shares a Notion page with others."
    parameters: List[ArgParameter] = [
        {
            "name": "page_id",
            "type": "string",
            "description": "The id of the page to be shared.",
            "required": True,
        },
        {
            "name": "recipients",
            "type": "array",
            "description": "The email addresses of the recipients. Each recipient should be a string.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "success",
            "type": "boolean",
            "description": "Whether the operation was successful.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'page_id' parameter is invalid, or the 'recipients' parameter is empty or contains invalid email addresses.",
        },
        {
            "name": "NotFoundException",
            "description": "The specified page does not exist.",
        },
    ]


class NotionManagerSearchContent(VirtualFunctionTool):
    name = "NotionManagerSearchContent"
    summary = "Searches for content in Notion."
    parameters: List[ArgParameter] = [
        {
            "name": "keywords",
            "type": "string",
            "description": "The keywords to search for.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "results",
            "type": "array",
            "description": "A list of objects. Each object contains 'id' (the id of the found content), 'title' (the title of the found content), and 'content' (the found content).",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'keywords' parameter is empty or invalid.",
        }
    ]


@register_toolkit()
class NotionManager(FunctionToolkit):
    name_for_human = "Notion Manager"
    description_for_human = "Toolkit for managing content in Notion."
    name_for_model = "NotionManager"
    description_for_model = "A toolkit for creating, organizing, and managing content in Notion. It includes tools for creating and editing pages and databases, managing tasks, embedding multimedia, integrating with other tools, sharing pages, and searching content."
    tool_classes = [
        NotionManagerCreatePage,
        NotionManagerReadPage,
        NotionManagerEditPage,
        NotionManagerSharePage,
        NotionManagerSearchContent,
    ]


#################### Messenger ####################


class MessengerSendMessage(VirtualFunctionTool):
    name = "MessengerSendMessage"
    summary = "Sends a text message to a specified user."
    parameters: List[ArgParameter] = [
        {
            "name": "recipient_id",
            "type": "string",
            "description": "The unique identifier of the recipient or the group chat.",
            "required": True,
        },
        {
            "name": "message",
            "type": "string",
            "description": "The content of the message.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "message_id",
            "type": "string",
            "description": "The unique identifier of the sent message.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'recipient_id' or 'message' is invalid or missing.",
        }
    ]


class MessengerReceiveMessage(VirtualFunctionTool):
    name = "MessengerReceiveMessage"
    summary = "Retrieves the latest messages for the user."
    parameters: List[ArgParameter] = [
        {
            "name": "max_count",
            "type": "integer",
            "description": "The maximum number of messages to retrieve. Default is 10.",
            "required": False,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "messages",
            "type": "array",
            "description": "A list of objects, each object contains 'message_id' (the unique identifier of the message), 'sender_id' (the unique identifier of the sender or the group chat), 'time' (the time of the message), 'message' (the content of the message).",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'max_count' is less than 1.",
        }
    ]


class MessengerSendMediaFile(VirtualFunctionTool):
    name = "MessengerSendMediaFile"
    summary = "Sends a media file to a specified user."
    parameters: List[ArgParameter] = [
        {
            "name": "recipient_id",
            "type": "string",
            "description": "The unique identifier of the recipient.",
            "required": True,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "The local path of the media file.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "file_id",
            "type": "string",
            "description": "The unique identifier of the sent file.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'recipient_id' or 'file_path' is invalid or missing.",
        }
    ]


class MessengerShareFile(VirtualFunctionTool):
    name = "MessengerShareFile"
    summary = "Shares a file with a specified user."
    parameters: List[ArgParameter] = [
        {
            "name": "recipient_id",
            "type": "string",
            "description": "The unique identifier of the recipient.",
            "required": True,
        },
        {
            "name": "file_path",
            "type": "string",
            "description": "The local path of the file.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "file_id",
            "type": "string",
            "description": "The unique identifier of the shared file.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'recipient_id' or 'file_path' is invalid or missing.",
        }
    ]


class MessengerSearchInChat(VirtualFunctionTool):
    name = "MessengerSearchInChat"
    summary = "Searches for a specified term in a chat."
    parameters: List[ArgParameter] = [
        {
            "name": "term",
            "type": "string",
            "description": "The term to search for.",
            "required": True,
        },
        {
            "name": "chat_id",
            "type": "string",
            "description": "The unique identifier of the chat.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "results",
            "type": "array",
            "description": "A list of objects, each object contains 'message_id' (the unique identifier of the message), 'message' (the content of the message).",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'term' or 'chat_id' is invalid or missing.",
        }
    ]


@register_toolkit()
class Messenger(FunctionToolkit):
    name_for_human = "Messenger"
    description_for_human = (
        "A toolkit for facilitating instant messaging and communication."
    )
    name_for_model = "Messenger"
    description_for_model = "A toolkit for facilitating instant messaging and communication. It enables users to send and receive messages, create group chats, send media files, react to messages with emojis, and search within chats."
    tool_classes = [
        MessengerSendMessage,
        MessengerReceiveMessage,
        MessengerSendMediaFile,
        MessengerShareFile,
        MessengerSearchInChat,
    ]


#################### ZoomManager ####################


class ZoomManagerGetMeetingTranscript(VirtualFunctionTool):
    name = "ZoomManagerGetMeetingTranscript"
    summary = "Retrieves the transcript of a specific Zoom meeting."
    parameters: List[ArgParameter] = [
        {
            "name": "meeting_id",
            "type": "string",
            "description": "The unique identifier of the Zoom meeting.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "transcript",
            "type": "string",
            "description": "The complete transcript of the Zoom meeting.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "NotFoundException",
            "description": "The 'meeting_id' does not exist or the meeting does not have a transcript.",
        }
    ]


class ZoomManagerSearchTranscript(VirtualFunctionTool):
    name = "ZoomManagerSearchTranscript"
    summary = (
        "Searches the transcript of a specific Zoom meeting for a keyword or phrase."
    )
    parameters: List[ArgParameter] = [
        {
            "name": "meeting_id",
            "type": "string",
            "description": "The unique identifier of the Zoom meeting.",
            "required": True,
        },
        {
            "name": "keyword",
            "type": "string",
            "description": "The keyword or phrase to search for in the transcript.",
            "required": True,
        },
    ]
    returns: List[ArgReturn] = [
        {
            "name": "search_results",
            "type": "array",
            "description": "The lines in the transcript where the keyword or phrase was found.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "NotFoundException",
            "description": "The 'meeting_id' does not exist or the meeting does not have a transcript.",
        }
    ]


class ZoomManagerGetParticipants(VirtualFunctionTool):
    name = "ZoomManagerGetParticipants"
    summary = "Retrieves the list of participants in a specific Zoom meeting."
    parameters: List[ArgParameter] = [
        {
            "name": "meeting_id",
            "type": "string",
            "description": "The unique identifier of the Zoom meeting.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "participants",
            "type": "array",
            "description": "The list of participants in the meeting.",
        }
    ]
    exceptions: List[ArgException] = [
        {"name": "NotFoundException", "description": "The 'meeting_id' does not exist."}
    ]


class ZoomManagerSearchMeetings(VirtualFunctionTool):
    name = "ZoomManagerSearchMeetings"
    summary = "Searches for Zoom meetings based on certain criteria."
    parameters: List[ArgParameter] = [
        {
            "name": "criteria",
            "type": "object",
            "description": "The criteria to search for meetings, including date, time, participants, or keywords in the meeting title or description.",
            "required": True,
        }
    ]
    returns: List[ArgReturn] = [
        {
            "name": "search_results",
            "type": "array",
            "description": "The list of meeting IDs that match the search criteria.",
        }
    ]
    exceptions: List[ArgException] = [
        {
            "name": "InvalidRequestException",
            "description": "The 'criteria' argument is malformed or contains erroneous parameters.",
        }
    ]


@register_toolkit()
class ZoomManager(FunctionToolkit):
    name_for_human = "Zoom Manager"
    description_for_human = "Toolkit for managing and interacting with Zoom meetings."
    name_for_model = "ZoomManager"
    description_for_model = "The ZoomManager toolkit is designed to manage and interact with Zoom meetings. It provides functionality to access and download transcripts and chat threads, search transcripts and chat threads for specific keywords, retrieve a list of participants and meeting details, download meeting recordings, and manage meeting settings."
    tool_classes = [
        ZoomManagerGetMeetingTranscript,
        ZoomManagerSearchTranscript,
        ZoomManagerGetParticipants,
        ZoomManagerSearchMeetings,
    ]
