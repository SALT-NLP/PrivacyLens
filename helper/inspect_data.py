import json

import streamlit as st


def load_json(uploaded_file):
    return json.load(uploaded_file)


def main():
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0
    st.title("Inspect PrivacyLens Data")

    uploaded_file = st.file_uploader("Choose the data file in JSON format", type="json")

    if uploaded_file:
        data = load_json(uploaded_file)
        names = [d['name'] for d in data]
        if st.session_state.selected_index >= len(data):
            st.session_state.selected_index = 0
        selected_option = st.selectbox('Select a case', names, index=st.session_state.selected_index)
        st.session_state.selected_index = names.index(selected_option)

        if selected_option:
            case_info = data[st.session_state.selected_index]
            seed = case_info['seed']
            st.markdown("## Privacy-Sensitive Seed\n"
                        f"- Data Type: {seed['data_type']}\n"
                        f"- Data Subject: {seed['data_subject']}\n"
                        f"- Data Sender: {seed['data_sender_name']}, {seed['data_sender']}\n"
                        f"- Data Recipient: {seed['data_recipient']}\n"
                        f"- Transmission Principle: {seed['transmission_principle']}\n"
                        f"- Source: {seed['source']}")

            st.write("## Vignette")
            vignette = case_info['vignette']
            st.info(vignette['story'])
            st.markdown(f"- Data Type (concrete): {vignette['data_type_concrete']}\n"
                        f"- Data Subject (concrete): {vignette['data_subject_concrete']}\n"
                        f"- Data Sender (concrete): {vignette['data_sender_concrete']}\n"
                        f"- Data Recipient (concrete): {vignette['data_recipient_concrete']}")

            trajectory = case_info['trajectory']
            st.markdown("## Trajectory\n"
                        "### Environment Information\n"
                        f"- User Name: {trajectory['user_name']}\n"
                        f"- User Email: {trajectory['user_email']}\n"
                        f"- User Instruction: {trajectory['user_instruction']}\n"
                        f"- Toolkits: {trajectory['toolkits']}")
            st.markdown("### Executable Trajectory")
            rounds = trajectory['executable_trajectory'].split('Action:')
            for r in rounds:
                if len(r) == 0:
                    continue
                try:
                    splits = r.split('Action Input:')
                    agent_action = splits[0].replace('Action:', '').strip()
                    splits = splits[1].split('Observation:')
                    action_input = splits[0].replace('Action Input:', '').strip()
                    observation = splits[1].replace('Observation:', '').strip()
                    observation = json.dumps(json.loads(observation), indent=2)
                    st.write(f"Agent: {agent_action}({action_input})")
                    st.write("Observation:")
                    st.json(observation)
                except Exception as e:
                    st.error(r)
            st.markdown("### Sensitive Information Items")
            items = [item.replace("$", "\\$") for item in trajectory['sensitive_info_items']]
            st.warning("\n".join(f'- {item}' for item in items))

        # Display navigation buttons
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    text-align: end;
                } 
            </style>
            """, unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Previous Case'):
                if st.session_state.selected_index > 0:
                    st.session_state.selected_index -= 1
                    st.rerun()
                else:
                    st.error("This is the first case.")
        with col2:
            if st.button('Next Case'):
                if st.session_state.selected_index < len(data) - 1:
                    st.session_state.selected_index += 1
                    st.rerun()
                else:
                    st.error("This is the last case.")


if __name__ == '__main__':
    main()
